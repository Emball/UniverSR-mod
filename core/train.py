import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import hashlib
import json
import logging
import shutil
import signal
import sys
import time
import warnings
from datetime import datetime

CORE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CORE)
sys.path.insert(0, CORE)

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf, open_dict  # noqa: E402
from pytorch_lightning.callbacks import Callback, ModelCheckpoint  # noqa: E402

from ckpt_utils import best_checkpoint, export_model, latest_checkpoint, list_checkpoints, load_ckpt, model_state_from_ckpt, rank_key  # noqa: E402
from data_prep import CACHE_DIR, prepare_data  # noqa: E402
from paired_datamodule import PairedAudioDataModule  # noqa: E402
from train_opt import apply_optimizations, freeze_by_prefix, make_optimizer, make_scheduler, resolve_precision  # noqa: E402
from universr.models.loader import build_model, load_pretrained, load_state_dict_into  # noqa: E402
from universr.sampling import make_transform  # noqa: E402
from universr.system import UniverSRSystem  # noqa: E402

warnings.filterwarnings("ignore")
log = logging.getLogger("universr.train")

MODELS_DIR = os.path.join(REPO_ROOT, "models")
PRETRAINED_NAMES = ("pytorch_model.bin", "universr.bin", "universr.pth", "universr.ckpt")
HF_REPO = "woongzip1/universr-audio"
SYSTEM_KEYS = {"sigma_min", "band_weights", "val_ode_steps", "val_guidance", "val_chunk_sec", "val_seed",
               "visqol_fraction", "keep_lq_below_cutoff"}


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    for noisy in ("matplotlib", "numba", "PIL", "urllib3", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def adopt_loose_files(name):
    data_root = os.path.join(REPO_ROOT, "data")
    if not os.path.isdir(data_root):
        return
    exts = {".wav", ".mp3", ".flac"}
    loose = [f for f in os.listdir(data_root)
             if os.path.isfile(os.path.join(data_root, f))
             and os.path.splitext(f)[0].upper().endswith(("_LQ", "_HQ"))
             and os.path.splitext(f)[1].lower() in exts]
    if not loose:
        return
    dst = os.path.join(data_root, name)
    os.makedirs(dst, exist_ok=True)
    for f in loose:
        shutil.move(os.path.join(data_root, f), os.path.join(dst, f))
    log.info("[autodiscovery] moved %d loose file(s) from data/ to data/%s/", len(loose), name)


def resolve_weights(cfg):
    path = cfg.get("weights_path")
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"weights_path not found: {path}")
        return path
    os.makedirs(MODELS_DIR, exist_ok=True)
    for fname in PRETRAINED_NAMES:
        cand = os.path.join(MODELS_DIR, fname)
        if os.path.isfile(cand):
            log.info("[weights] found pretrained model in models/: %s", fname)
            return cand
    try:
        from huggingface_hub import hf_hub_download
        log.info("[weights] none in models/ -- downloading %s from Hugging Face", HF_REPO)
        return hf_hub_download(repo_id=HF_REPO, filename="pytorch_model.bin")
    except Exception as e:
        raise FileNotFoundError(
            f"No pretrained weights found. Place pytorch_model.bin from {HF_REPO} in models/ "
            f"or set weights_path (Hugging Face download failed: {e})") from e


def load_weights(model, path):
    if path.endswith(".ckpt"):
        sd = model_state_from_ckpt(load_ckpt(path))
        log.info("[weights] Lightning checkpoint (EMA applied if present): %s", os.path.basename(path))
        load_state_dict_into(model, sd)
    else:
        load_pretrained(model, path)
    log.info("[weights] loaded")


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def baseline_key(weights_path, eval_dir, model, system, cfg):
    payload = {
        "weights": file_md5(weights_path) if weights_path and os.path.isfile(weights_path) else "",
        "val_key": os.path.basename(os.path.dirname(os.path.normpath(eval_dir))),
        "gen_start": int(model.gen_start_bin), "aligned": bool(model.aligned_input), "anchors": list(model.bw_anchor_bins),
        "keep_lq": bool(system.keep_lq), "songs": system.val_songs, "steps": system.val_ode_steps,
        "guidance": system.val_guidance, "seed": system.val_seed, "visqol_fraction": system.visqol_fraction,
        "sr": int(cfg.datas.sr), "cutoff": OmegaConf.to_container(cfg.datas, resolve=True).get("cutoff_hz", "auto"),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def run_baseline(trainer, system, datamodule, key):
    cache_dir = os.path.join(CACHE_DIR, "baseline")
    cache_file = os.path.join(cache_dir, f"{key}.json")
    if os.path.isfile(cache_file):
        try:
            with open(cache_file) as f:
                bl = json.load(f)
            log.info("[baseline] cached (%s)", key[:8])
        except Exception as e:
            log.warning("[baseline] cache read failed (%s) -- re-running", e)
            os.remove(cache_file)
            bl = None
    else:
        bl = None
    if bl is None:
        log.info("[baseline] evaluating pretrained weights")
        datamodule.setup("fit")
        res = trainer.validate(system, datamodule=datamodule, verbose=False)
        if not res:
            return
        bl = {k: float(v) for k, v in res[0].items() if v is not None}
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(bl, f, indent=2)
    system._last_val_sisdr = bl.get("sisdr")
    system._last_val_hfnr = bl.get("hfnr")
    system._last_val_visqol = bl.get("visqol")
    system._last_val_lsd = bl.get("lsd_high")
    log.info("[baseline] %s", "  ".join(f"{k}={v:.3f}" for k, v in bl.items() if k in ("visqol", "sisdr", "hfnr", "lsd_high")))


def reconcile_resume_ckpt(ckpt_path, system, run_dir):
    # a changed config (arm, region, anchors) makes tensors mismatch; fill from the fresh model and restart optimizers
    ck = load_ckpt(ckpt_path)
    live = system.state_dict()
    sd = ck["state_dict"]
    drop = [k for k in sd if k not in live]
    fill = [k for k in live if k not in sd]
    reshape = [k for k in sd if k in live and sd[k].shape != live[k].shape]
    if not (drop or fill or reshape):
        return ckpt_path
    for k in drop + reshape:
        sd.pop(k, None)
    for k in fill + reshape:
        sd[k] = live[k]
    ck["optimizer_states"], ck["lr_schedulers"] = [], []
    out = os.path.join(run_dir, ".resume_reconciled.ckpt")
    torch.save(ck, out)
    log.warning("[resume] config differs from checkpoint: dropped %d, filled %d, reshaped %d tensors; "
                "optimizers and schedule restart", len(drop), len(fill), len(reshape))
    return out


class StepPrinter(Callback):
    def __init__(self, base_dir, run_dir, interrupt):
        self.base_dir, self.run_dir, self.interrupt = base_dir, run_dir, interrupt
        self.total, self.last_step, self.last_idx = 0, 0, 0
        self.t0, self.done = None, 0
        self.val_elapsed, self.val_t0 = 0.0, None
        self.sanity = True

    def on_train_epoch_start(self, trainer, pl_module):
        self.total = trainer.num_training_batches
        self.last_step = trainer.global_step
        self.last_idx = 0
        self.t0 = None
        self.done = 0
        self.val_elapsed = 0.0
        self.val_t0 = None
        print(f"\nEpoch {trainer.current_epoch} -- {self.total} batches", flush=True)

    def _vals(self, m):
        parts = []
        for key, attr, fmt in (("visqol", "_last_val_visqol", "{:.3f}"), ("sisdr", "_last_val_sisdr", "{:.3f}"),
                               ("hfnr", "_last_val_hfnr", "{:.3f}"), ("lsd", "_last_val_lsd", "{:.3f}")):
            v = getattr(m, attr, None)
            if v is not None:
                parts.append(f"{key}=" + fmt.format(float(v)))
        return ("  " + "  ".join(parts)) if parts else ""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
        self.done += 1
        elapsed = (now - self.t0) - self.val_elapsed
        its = self.done / elapsed if elapsed > 0 else 0.0
        if self.interrupt["requested"]:
            save_interrupt(trainer, self.run_dir)
        if trainer.global_step == self.last_step:
            return
        self.last_step = trainer.global_step
        self.last_idx = batch_idx
        pct = 100 * (batch_idx + 1) / max(1, self.total)
        print(f"\r  {pct:5.1f}%  step={trainer.global_step}  {batch_idx + 1}/{self.total}  {its:.2f} it/s"
              f"{self._vals(pl_module)}", end="", flush=True)
        self.pause_check()

    def pause_check(self):
        flag = next((p for p in (os.path.join(self.base_dir, "PAUSED"), os.path.join(self.run_dir, "PAUSED"))
                     if os.path.isfile(p)), None)
        if flag is None:
            return
        print("\n[paused] inference in progress -- training suspended", flush=True)
        t = time.monotonic()
        while os.path.isfile(flag):
            time.sleep(0.5)
        self.val_elapsed += time.monotonic() - t
        print("[resumed] training continuing", flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        print(flush=True)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_t0 = time.monotonic()
        self.sanity = trainer.sanity_checking
        if not self.sanity:
            print("\r  Validating...                                                  ", end="", flush=True)

    def on_validation_end(self, trainer, pl_module):
        if self.val_t0 is not None:
            self.val_elapsed += time.monotonic() - self.val_t0
        if not getattr(self, "sanity", True) and getattr(self, "total", 0):
            pct = 100 * (self.last_idx + 1) / self.total
            print(f"\r  {pct:5.1f}%  step={trainer.global_step}  {self.last_idx + 1}/{self.total}  --{self._vals(pl_module)}",
                  end="", flush=True)


def save_interrupt(trainer, run_dir):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    out = os.path.join(ckpt_dir, f"step={trainer.global_step:06d}.ckpt")
    try:
        trainer.save_checkpoint(out)
        log.info("\n[interrupt] saved %s", out)
    except Exception as e:
        log.error("\n[interrupt] save failed: %s", e)
    sys.stdout.flush()
    os._exit(0)


def install_interrupt_handler(interrupt):
    def handler(sig=None, frame=None):
        if interrupt["requested"]:
            log.info("\n[interrupt] second Ctrl+C -- quitting without saving")
            os._exit(130)
        interrupt["requested"] = True
        log.info("\n[interrupt] Ctrl+C caught -- saving at the next training step (press again to quit without saving)")
    signal.signal(signal.SIGINT, handler)


def pick_run_dir(cfg, base_dir):
    ckpt_path = None
    if cfg.get("resume", False) and os.path.isdir(base_dir):
        subs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d, "checkpoints"))]
        if subs:
            run_dir = max(subs, key=os.path.getmtime)
            ckpt_path = latest_checkpoint(os.path.join(run_dir, "checkpoints"))
            if ckpt_path:
                log.info("[resume] run %s -- checkpoint %s", os.path.basename(run_dir), os.path.basename(ckpt_path))
            else:
                log.info("[resume] run %s has no checkpoints -- starting from pretrained weights", os.path.basename(run_dir))
            return run_dir, ckpt_path
        log.info("[resume] no existing runs -- starting fresh")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_id)
    log.info("[run] new run: %s", run_id)
    return run_dir, ckpt_path


def build_everything(cfg, run_dir, resuming):
    mc = OmegaConf.to_container(cfg.model, resolve=True)
    tcfg = OmegaConf.to_container(cfg.transform, resolve=True)
    sr = int(cfg.datas.sr)
    if int(tcfg["sampling_rate"]) != sr:
        raise ValueError(f"datas.sr ({sr}) must equal transform.sampling_rate ({tcfg['sampling_rate']}); "
                         "the pretrained weights are 48 kHz")
    model = build_model(mc, gen_start_bin=mc.get("gen_start_bin"), aligned_input=bool(mc.get("aligned_input", False)),
                        bw_anchor_bins=mc.get("bw_anchor_bins"), grad_checkpoint=bool(mc.get("grad_checkpoint", False)))
    weights = None
    if resuming:
        log.info("[weights] resuming -- checkpoint supplies weights")
    else:
        weights = resolve_weights(cfg)
        load_weights(model, weights)

    tr = cfg.training
    freeze_by_prefix(model, list(tr.get("freeze", []) or []))
    trainable = [p for p in model.parameters() if p.requires_grad]
    log.info("[model] trainable params: %.2fM", sum(p.numel() for p in trainable) / 1e6)

    optimizer = make_optimizer(trainable, OmegaConf.to_container(cfg.optimizer, resolve=True))
    scheduler = make_scheduler(optimizer, OmegaConf.to_container(cfg.get("scheduler", {}), resolve=True), int(tr.max_steps))

    sc = OmegaConf.to_container(cfg.get("system", {}), resolve=True) or {}
    unknown = set(sc) - SYSTEM_KEYS
    if unknown:
        log.warning("[config] ignoring unknown system keys: %s", sorted(unknown))
    system = UniverSRSystem(
        model, make_transform(tcfg), optimizer=optimizer, scheduler=scheduler, sample_rate=sr,
        ema_decay=float(tr.get("ema_decay", 0.0)), val_songs=tr.get("val_songs", 3),
        val_rotate_every=tr.get("val_rotate_every", "auto"),
        val_audio_dir=os.path.join(run_dir, "val_audio"), **{k: v for k, v in sc.items() if k in SYSTEM_KEYS})
    return model, system, weights, mc, tcfg


def train(cfg: DictConfig):
    apply_optimizations(cfg.get("optimizations", {}), REPO_ROOT)
    if cfg.get("seed") is not None:
        pl.seed_everything(int(cfg.seed), workers=True)

    prepare_data(cfg)
    train_lq = os.path.join(cfg.datas.train_dir, "LQ")
    if not os.path.isdir(train_lq) or not any(f.endswith(".wav") for f in os.listdir(train_lq)):
        n = cfg.exp.name
        log.error("ERROR: no training chunks found. Populate data/%s/{train,val}/{LQ,HQ} with paired audio "
                  "(same filenames in LQ and HQ), or drop _LQ/_HQ files in data/.", n)
        raise SystemExit(1)

    dm_cfg = {k: v for k, v in OmegaConf.to_container(cfg.datas, resolve=True).items() if k != "_target_"}
    datamodule = PairedAudioDataModule(**dm_cfg)

    base_dir = os.path.join(cfg.exp.dir, cfg.exp.name)
    run_dir, ckpt_path = pick_run_dir(cfg, base_dir)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    model, system, weights, mc, tcfg = build_everything(cfg, run_dir, resuming=ckpt_path is not None)

    tr = cfg.training
    accum = int(tr.get("grad_accum_steps", 1))
    precision = resolve_precision(tr.get("precision", "auto"))
    accel = "gpu" if torch.cuda.is_available() else "cpu"
    log.info("[trainer] accelerator=%s precision=%s accum=%d clip=%s max_steps=%d", accel, precision, accum,
             tr.get("grad_clip", 1.0), int(tr.max_steps))

    interrupt = {"requested": False}
    callbacks = [StepPrinter(base_dir, run_dir, interrupt)]
    val_disabled = float(tr.get("limit_val_batches", 1.0)) == 0.0
    checkpoint = None
    if not val_disabled:
        checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir, monitor="visqol", mode="max", save_top_k=-1, save_last=False, auto_insert_metric_name=True,
            filename="{step:06d}-{sisdr:.3f}-{visqol:.3f}-{hfnr:.3f}")
        callbacks.append(checkpoint)

    logger = False
    try:
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir=os.path.join(run_dir, "logs"), name="", version="")
    except Exception as e:
        log.warning("[logger] TensorBoard unavailable (%s) -- continuing without it", e)

    vci = tr.get("val_check_interval", 500)
    datamodule.setup("fit")
    n_batches = len(datamodule.data_train) // int(cfg.datas.get("batch_size", 1))
    limit_train = (n_batches // accum) * accum if accum > 1 and n_batches >= accum else 1.0
    if limit_train != 1.0:
        log.info("[trainer] %d batches/epoch trimmed to %d so accumulation windows never straddle epochs "
                 "(step counts stay aligned with val_check_interval)", n_batches, limit_train)
    trainer = pl.Trainer(
        max_steps=int(tr.max_steps), accelerator=accel, devices=1, precision=precision,
        accumulate_grad_batches=accum, gradient_clip_val=float(tr.get("grad_clip", 1.0)) or None,
        val_check_interval=int(vci) * accum, check_val_every_n_epoch=None, limit_train_batches=limit_train,
        limit_val_batches=float(tr.get("limit_val_batches", 1.0)), num_sanity_val_steps=0,
        callbacks=callbacks, logger=logger, enable_progress_bar=False, enable_model_summary=False,
        default_root_dir=run_dir, log_every_n_steps=int(tr.get("log_every_n_steps", 10)))

    if ckpt_path is None and not val_disabled:
        try:
            run_baseline(trainer, system, datamodule, baseline_key(weights, cfg.datas.eval_dir, model, system, cfg))
        except Exception as e:
            log.warning("[baseline] skipped: %s", e)
    if ckpt_path is not None:
        ckpt_path = reconcile_resume_ckpt(ckpt_path, system, run_dir)

    install_interrupt_handler(interrupt)
    try:
        trainer.fit(system, datamodule=datamodule, ckpt_path=ckpt_path)
    except torch.cuda.OutOfMemoryError as e:
        log.error("\n[OOM] CUDA out of memory -- exiting cleanly. Reduce batch_size, enable model.grad_checkpoint, "
                  "or raise training.grad_accum_steps.\n%s", e)
        torch.cuda.empty_cache()
        os._exit(1)
    except MemoryError as e:
        log.error("\n[OOM] system RAM exhausted -- exiting cleanly.\n%s", e)
        os._exit(1)
    log.info("Training finished!")

    items = list_checkpoints(ckpt_dir)
    with open(os.path.join(run_dir, "best_k_models.json"), "w") as f:
        json.dump({os.path.basename(i["path"]): {k: i[k] for k in ("step", "sisdr", "visqol", "hfnr")}
                   for i in sorted(items, key=rank_key, reverse=True)}, f, indent=1)
    best = best_checkpoint(ckpt_dir)
    if best:
        export_model(best, os.path.join(run_dir, "best_model.pth"), model, mc, tcfg)
        log.info("[train] best model: %s -> %s/best_model.pth", os.path.basename(best), run_dir)
    else:
        log.info("[train] no evaluated checkpoint found -- skipping best_model.pth export")


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf_dir", default="configs/universr.yaml", help="path to config file")
    ap.add_argument("--weights_path", default=None, help="pretrained weights (.bin/.pth) or a previous .ckpt")
    ap.add_argument("--resume", action="store_true", help="resume the latest run's newest checkpoint")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.conf_dir)
    stem = os.path.splitext(os.path.basename(args.conf_dir))[0]
    with open_dict(cfg):
        if not cfg.get("exp"):
            cfg.exp = {}
        if not cfg.exp.get("dir"):
            cfg.exp.dir = os.path.join(REPO_ROOT, "runs")
        if not cfg.exp.get("name"):
            cfg.exp.name = stem
        if args.weights_path:
            cfg.weights_path = args.weights_path
        if args.resume:
            cfg.resume = True
    log.info("[autodiscovery] name=%s data=data/%s", cfg.exp.name, cfg.exp.name)
    adopt_loose_files(cfg.exp.name)
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    main()
