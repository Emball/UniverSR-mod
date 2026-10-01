import logging
import math
import os
import random
import time
import re

import pytorch_lightning as pl
import torch

from universr import diagnose, memlog, metrics as M
from universr.flow.loss import band_weight_vector, flow_matching_loss
from universr.flow.path import OriginalCFMPath
from universr.sampling import hz_to_cutoff_bins, restore_long, to_spec

log = logging.getLogger("universr.system")

_CLIP_SUFFIX = re.compile(r"_\d{4}$")


def _song_key(path):
    return _CLIP_SUFFIX.sub("", os.path.splitext(os.path.basename(path))[0])


class UniverSRSystem(pl.LightningModule):
    def __init__(
        self,
        model,
        transform,
        optimizer=None,
        scheduler=None,
        sigma_min=1e-4,
        sample_rate=48000,
        band_weights=None,
        ema_decay=0.0,
        val_songs=3,
        val_rotate_every="auto",
        val_ode_steps=4,
        val_guidance=1.5,
        val_chunk_sec=None,
        val_audio_dir=None,
        val_seed=1234,
        visqol_fraction=1.0,
        keep_lq_below_cutoff="auto",
        lsd_cutoff_hz=None,
        mem_log=False,
        mem_log_every=50,
        t_skew=1.0,
        aux_logmag_weight=0.0,
        alpha_start=None,
        alpha_anneal_steps=0,
    ):
        super().__init__()
        self.audio_model = model
        self.transform = transform
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.path = OriginalCFMPath(sigma_min=sigma_min)
        self.sample_rate = sample_rate
        self.n_fft = transform.complex_stft.n_fft
        self.ema_decay = float(ema_decay)
        self.val_songs = max(1, int(val_songs))
        self.val_rotate_every = val_rotate_every
        self.val_ode_steps = int(val_ode_steps)
        self.val_guidance = val_guidance
        self.val_chunk_sec = val_chunk_sec
        self.val_audio_dir = val_audio_dir
        self.val_seed = int(val_seed)
        self.mem_log = bool(mem_log)
        self.t_skew = float(t_skew)
        self.aux_logmag_weight = float(aux_logmag_weight)
        self.alpha_start = None if alpha_start is None else float(alpha_start)
        self.alpha_anneal_steps = int(alpha_anneal_steps or 0)
        self._final_alpha = float(transform.compress.compression_exponent)
        self._alpha_origin = None
        self.mem_log_every = int(mem_log_every)
        self.val_step_override = None
        self._chunk_t = None
        self.visqol_fraction = max(0.0, min(1.0, float(visqol_fraction)))
        self.lsd_cutoff_hz = None if lsd_cutoff_hz is None else float(lsd_cutoff_hz)
        self.keep_lq = (not model.aligned_input) if keep_lq_below_cutoff == "auto" else bool(keep_lq_below_cutoff)

        bin_hz = sample_rate / self.n_fft
        self.register_buffer(
            "band_w",
            band_weight_vector(band_weights, model.gen_start_bin, model.hr_freq_bins, bin_hz),
            persistent=False,
        )

        self._diag_done = False
        self._bad_steps = 0
        self._last_ctx = None

        self._ema = None
        self._ema_pending = None
        self._ema_last_step = -1
        self._ema_active = False

        self._val_all_songs = []
        self._val_saved_keys = None
        self._val_window_idx = 0
        self._val_next_rotate = -1
        self._val_rotate_steps = None
        self._val_window_best = {}
        self._active_indices = set()
        self._val_rows = []

        self._last_val_sisdr = None
        self._last_val_hfnr = None
        self._last_val_visqol = None
        self._last_val_lsd = None
        self._baseline = None

    def forward(self, x, t, y, cutoff_bins):
        return self.audio_model(x, t, y, cutoff_bins)

    def _bins(self, cutoff_hz):
        return hz_to_cutoff_bins(cutoff_hz, self.sample_rate, self.n_fft).to(self.device).reshape(-1)

    def current_alpha(self):
        if self.alpha_start is None or self.alpha_anneal_steps <= 0:
            return self._final_alpha
        if self._alpha_origin is None:
            return self.alpha_start
        f = min(1.0, max(0.0, (self.global_step - self._alpha_origin) / self.alpha_anneal_steps))
        return self.alpha_start + (self._final_alpha - self.alpha_start) * f

    def _apply_alpha(self):
        self.transform.compress.compression_exponent = self.current_alpha()

    def _cfm_loss(self, hq, lq, cutoff_hz, t=None, generator=None, train=False):
        self._apply_alpha()
        Z = to_spec(self.transform, hq)
        Y = to_spec(self.transform, lq)
        Z_hr = Z[:, :, self.audio_model.gen_start_bin:]
        B = Z.shape[0]
        if t is None:
            t = torch.rand(B, 1, 1, 1, device=Z.device)
            if self.t_skew != 1.0:
                t = 1.0 - (1.0 - t) ** self.t_skew
        if generator is None:
            x0 = self.path.sample_source(Z_hr)
        else:
            x0 = torch.randn(Z_hr.shape, device=Z.device, generator=generator)
        xt = self.path.sample_xt(x0, Z_hr, t)
        bins = self._bins(cutoff_hz)
        self._last_ctx = (xt.detach(), t.detach(), Y.detach(), bins, {"hq": hq, "lq": lq, "Z": Z})
        out = self.audio_model(xt, t, Y, bins)
        target = self.path.get_target_vector_field(xt, x0, Z_hr, t)
        loss = flow_matching_loss(out, target, self.band_w)
        if train and self.aux_logmag_weight > 0:
            x1_hat = xt.float() + (1.0 - t) * out.float()
            m_hat = x1_hat.square().sum(1).add(1e-12).sqrt()
            m_true = Z_hr.float().square().sum(1).add(1e-12).sqrt()
            d = (torch.log(m_hat + 1e-6) - torch.log(m_true + 1e-6)) / self.current_alpha()
            aux = (d.square() * t.reshape(-1, 1, 1).square()).mean()
            self.log("train_aux", aux.detach(), on_step=True, logger=True, batch_size=hq.shape[0])
            loss = loss + self.aux_logmag_weight * aux
        return loss

    def training_step(self, batch, batch_idx):
        hq, lq, cutoff_hz = batch
        loss = self._cfm_loss(hq, lq, cutoff_hz, train=True)
        self._check_finite(loss)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True, batch_size=hq.shape[0])
        return loss

    def _check_finite(self, loss):
        if bool(torch.isfinite(loss.detach())):
            self._bad_steps = 0
            return
        self._bad_steps += 1
        if not self._diag_done and self._last_ctx is not None:
            self._diag_done = True
            xt, t, Y, bins, extra = self._last_ctx
            log.warning("[nan-diag] non-finite loss at step %d, tracing one forward", self.global_step)
            diagnose.run(self.audio_model, xt, t, Y, bins, extra)
        if self._bad_steps >= 25:
            raise RuntimeError("25 consecutive non-finite losses; see [nan-diag] lines above")

    def _ensure_ema(self):
        if self.ema_decay <= 0:
            return
        with torch.inference_mode(False), torch.no_grad():
            if self._ema is None:
                self._ema = {n: p.detach().clone() for n, p in self.audio_model.named_parameters()}
            for n, t in list(self._ema.items()):
                if t.is_inference():
                    self._ema[n] = t.clone()
            if self._ema_pending is not None:
                for n, v in self._ema_pending.items():
                    if n in self._ema and self._ema[n].shape == v.shape:
                        self._ema[n].copy_(v.to(self._ema[n].device))
                log.info("EMA restored from checkpoint")
                self._ema_pending = None

    def on_train_start(self):
        if self.alpha_start is not None and self.alpha_anneal_steps > 0:
            if self._alpha_origin is None:
                self._alpha_origin = int(self.global_step)
            log.info("[alpha] anneal %.3f -> %.3f over %d steps from step %d (now %.4f)", self.alpha_start,
                     self._final_alpha, self.alpha_anneal_steps, self._alpha_origin, self.current_alpha())
        if self.ema_decay <= 0:
            return
        self._ensure_ema()
        self._ema_last_step = self.global_step
        log.info("EMA enabled, decay=%.5f", self.ema_decay)
        if self.mem_log:
            memlog.snap("train start", reset_peak=True)

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.mem_log and self.mem_log_every > 0 and (self.global_step % self.mem_log_every == 0) and batch_idx % max(1, self.trainer.accumulate_grad_batches) == 0:
            memlog.snap(f"train step {self.global_step}", reset_peak=True)
        if self._ema is None or self.global_step == self._ema_last_step:
            return
        self._ema_last_step = self.global_step
        d = min(self.ema_decay, (1 + self.global_step) / (10 + self.global_step))
        for n, p in self.audio_model.named_parameters():
            self._ema[n].mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def _ema_swap(self):
        for n, p in self.audio_model.named_parameters():
            tmp = p.detach().clone()
            p.copy_(self._ema[n])
            self._ema[n].copy_(tmp)

    def _ema_use(self, active):
        if self._ema is None or self._ema_active == active:
            return
        self._ema_swap()
        self._ema_active = active

    def _amp_dtype(self):
        p = str(getattr(self.trainer, "precision", ""))
        if "bf16" in p:
            return torch.bfloat16
        return torch.float16 if "16" in p else None

    def on_validation_start(self):
        self._apply_alpha()
        if self.mem_log:
            memlog.snap("val start (before ema/empty)", reset_peak=True)
        if self._ema is None and self._ema_pending is not None:
            self._ensure_ema()
        self._ema_use(True)
        self._cudnn_bench = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.mem_log:
            memlog.snap("val start (after empty_cache)", reset_peak=True)

    def on_validation_end(self):
        self._ema_use(False)
        torch.backends.cudnn.benchmark = getattr(self, "_cudnn_bench", torch.backends.cudnn.benchmark)
        if self.mem_log:
            memlog.snap("val end (before empty_cache)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.mem_log:
            memlog.snap("val end (after empty_cache)", reset_peak=True)

    def _val_dataset(self):
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and getattr(dm, "data_val", None) is not None:
            return dm.data_val
        dl = self.trainer.val_dataloaders
        return (dl[0] if isinstance(dl, (list, tuple)) else dl).dataset

    def _lock_val_songs(self):
        ds = self._val_dataset()
        by_song = {}
        for i, (lq_path, _) in enumerate(ds.pairs):
            by_song.setdefault(_song_key(lq_path), []).append(i)
        keys = sorted(by_song)
        if self._val_saved_keys:
            kept = [k for k in self._val_saved_keys if k in by_song]
            keys = kept + [k for k in keys if k not in kept]
        self._val_all_songs = [(k, by_song[k]) for k in keys]
        steps = self.val_rotate_every
        if steps in ("auto", None):
            max_steps = self.trainer.max_steps
            windows = math.ceil(len(keys) / self.val_songs)
            steps = max(1, max_steps // windows) if max_steps and max_steps > 0 and windows > 1 else None
        else:
            steps = int(steps)
        self._val_rotate_steps = steps
        self._apply_val_window()
        log.info("val songs locked: %d songs, %d per window, rotate every %s steps",
                 len(keys), self.val_songs, steps)

    def _apply_val_window(self):
        n = len(self._val_all_songs)
        k = min(self.val_songs, n)
        start = (self._val_window_idx * k) % n if n else 0
        chosen = [self._val_all_songs[(start + i) % n] for i in range(k)] if n else []
        self._active_indices = {i for _, idxs in chosen for i in idxs}
        self._active_songs = [key for key, _ in chosen]

    def on_validation_epoch_start(self):
        self._val_rows = []
        if not self._val_all_songs:
            self._lock_val_songs()
        if self.trainer.sanity_checking or not self._val_rotate_steps or len(self._val_all_songs) <= self.val_songs:
            return
        step = self.global_step
        if self._val_next_rotate < 0:
            self._val_next_rotate = step + self._val_rotate_steps
        elif step >= self._val_next_rotate:
            best = self._val_window_best
            if best:
                parts = [f"{k}={best[k]:.3f}" for k in ("visqol", "hfnr") if k in best]
                print(f"[val] Window {self._val_window_idx + 1} best: {' '.join(parts)}  -- rotating songs")
            self._val_window_idx += 1
            self._val_window_best = {}
            self._val_next_rotate = step + self._val_rotate_steps
            self._apply_val_window()

    def _chunk_progress(self, done, total):
        now = time.time()
        dt = "" if self._chunk_t is None else f" chunk {now - self._chunk_t:.1f}s"
        self._chunk_t = now
        memlog.snap(f"  restore {done}/{total}{dt}")

    def _restore(self, lq, cutoff_hz, seed, steps=None):
        self._chunk_t = time.time() if self.mem_log else None
        return restore_long(
            self.audio_model, self.transform, lq, cutoff_hz, self.sample_rate,
            chunk_sec=self.val_chunk_sec,
            steps=self.val_ode_steps if steps is None else steps,
            guidance=self.val_guidance,
            keep_lq_below_cutoff=self.keep_lq,
            seed=seed,
            amp_dtype=self._amp_dtype(),
            progress=self._chunk_progress if self.mem_log else None,
        )

    def validation_step(self, batch, batch_idx):
        hq, lq, idx, song_key, cutoff_hz = batch
        idx = int(idx[0])
        song_key = song_key[0] if isinstance(song_key, (list, tuple)) else song_key

        if self.trainer.sanity_checking:
            if batch_idx == 0:
                self._restore(lq, cutoff_hz, self.val_seed, steps=1)
                log.info("sanity: restored a %.1f s clip", lq.shape[-1] / self.sample_rate)
            return None
        if idx not in self._active_indices:
            return None

        t0 = time.time()
        if self.mem_log:
            memlog.snap(f"clip {song_key} begin")
        est = self._restore(lq, cutoff_hz, self.val_seed + idx).clamp(-1.0, 1.0)
        if hq.is_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        if self.mem_log:
            memlog.snap(f"clip {song_key} sampled")
        if not torch.isfinite(est).all():
            log.warning("val clip %s: restored audio is non-finite (fp16 overflow?) -- its metrics are empty", song_key)
        gen = torch.Generator(device=hq.device)
        gen.manual_seed(self.val_seed + idx)
        n = hq.shape[-1]
        crop = int((self.val_chunk_sec or 4.0) * self.sample_rate)
        lo = max(0, (n - crop) // 2)
        cfm = float(self._cfm_loss(hq[..., lo:lo + crop], lq[..., lo:lo + crop], cutoff_hz,
                                   t=torch.full((1, 1, 1, 1), 0.5, device=hq.device), generator=gen))

        row = {
            "sisdr": M.sisdr(est, hq),
            "hfnr": M.hfnr(est, hq, self.sample_rate),
            "lsd_high": M.lsd(est, hq, self.lsd_cutoff_hz or float(cutoff_hz[0]), self.sample_rate)[1],
            "cfm": cfm,
            "visqol": None,
            "bands": M.band_db(est, hq, self.sample_rate),
        }
        t2 = time.time()
        if self.mem_log:
            memlog.snap(f"clip {song_key} metrics done")
        if random.Random(self.val_seed + idx).random() < self.visqol_fraction:
            row["visqol"] = M.visqol(est, hq, self.sample_rate)
        self._val_rows.append(row)
        log.info("val clip %s: sample %.1fs, metrics %.1fs, visqol %.1fs", song_key, t1 - t0, t2 - t1, time.time() - t2)

        if self.val_audio_dir:
            out = os.path.join(self.val_audio_dir, f"step_{self.global_step if self.val_step_override is None else self.val_step_override:06d}")
            os.makedirs(out, exist_ok=True)
            from audio_io import save_wav_f32
            for tag, wav in (("LQ", lq), ("HQ", hq), ("Restored", est)):
                save_wav_f32(wav[0].cpu(), os.path.join(out, f"{song_key}_{tag}.wav"), self.sample_rate)
        return None

    def on_validation_epoch_end(self):
        # live weights must be restored here: checkpoint callbacks run their on_validation_end before the module's
        try:
            self._val_epoch_end()
        finally:
            self._ema_use(False)

    def _val_epoch_end(self):
        def mean(key):
            vals = [r[key] for r in self._val_rows if r[key] is not None and not math.isnan(r[key])]
            return sum(vals) / len(vals) if vals else None

        if self.trainer.sanity_checking:
            return
        sisdr, hfnr, visqol, lsd_high, cfm = (mean(k) for k in ("sisdr", "hfnr", "visqol", "lsd_high", "cfm"))
        self._last_val_sisdr, self._last_val_hfnr = sisdr, hfnr
        self._last_val_visqol, self._last_val_lsd = visqol, lsd_high

        if visqol is not None and visqol > self._val_window_best.get("visqol", float("-inf")):
            self._val_window_best["visqol"] = visqol
        if hfnr is not None and hfnr < self._val_window_best.get("hfnr", float("inf")):
            self._val_window_best["hfnr"] = hfnr

        self.log("sisdr", float(sisdr) if sisdr is not None else -100.0, prog_bar=True, logger=True)
        self.log("hfnr", float(hfnr) if hfnr is not None else 0.0, logger=True)
        self.log("visqol", float(visqol) if visqol is not None else -1.0, logger=True)
        self.log("lsd_high", float(lsd_high) if lsd_high is not None else 0.0, logger=True)
        self.log("val_cfm", float(cfm) if cfm is not None else 0.0, logger=True)
        bands = []
        for i in range(len(M.BAND_LABELS)):
            vals = [r["bands"][i] for r in self._val_rows if not math.isnan(r["bands"][i])]
            bands.append(sum(vals) / len(vals) if vals else None)
            if bands[-1] is not None:
                self.log(f"band_db_{i}", float(bands[-1]), logger=True)
        base = self._baseline or {}
        parts = [f"{lab}={v:+.1f}" + (f" (base {base[f'band_db_{i}']:+.1f})" if f"band_db_{i}" in base else "")
                 for i, (lab, v) in enumerate(zip(M.BAND_LABELS, bands)) if v is not None]
        if parts:
            print(f"\n[bands] restored minus HQ, dB:  {'  '.join(parts)}  (alpha {self.current_alpha():.3f})", flush=True)
        if self.optimizer is not None:
            self.log("lr", self.optimizer.param_groups[0]["lr"], prog_bar=True)
        log.info("val: %d clips, songs=%s", len(self._val_rows), getattr(self, "_active_songs", []))

    def on_save_checkpoint(self, checkpoint):
        checkpoint["alpha"] = float(self.current_alpha())
        checkpoint["alpha_origin"] = self._alpha_origin
        checkpoint["val_song_keys"] = [k for k, _ in self._val_all_songs]
        checkpoint["val_window_idx"] = self._val_window_idx
        checkpoint["val_next_rotate"] = self._val_next_rotate
        checkpoint["val_window_best"] = self._val_window_best
        if self._ema is not None:
            checkpoint["ema"] = {n: v.detach().cpu() for n, v in self._ema.items()}

    def on_load_checkpoint(self, checkpoint):
        self._alpha_origin = checkpoint.get("alpha_origin")
        self._val_saved_keys = checkpoint.get("val_song_keys")
        self._val_window_idx = checkpoint.get("val_window_idx", 0)
        self._val_next_rotate = checkpoint.get("val_next_rotate", -1)
        self._val_window_best = checkpoint.get("val_window_best", {})
        self._ema_pending = checkpoint.get("ema")
        self._val_all_songs = []

    def configure_optimizers(self):
        if self.optimizer is None:
            raise RuntimeError("UniverSRSystem needs an optimizer to train")
        if self.scheduler is None:
            return self.optimizer
        sched = self.scheduler if isinstance(self.scheduler, dict) else {"scheduler": self.scheduler, "interval": "step"}
        return [self.optimizer], [sched]
