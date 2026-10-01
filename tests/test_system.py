import logging
import math
import os
import sys
import tempfile

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.join(os.path.dirname(__file__), "..", "core")
sys.path.insert(0, ROOT)

from universr.models.loader import build_model  # noqa: E402
from universr.sampling import make_transform  # noqa: E402
from universr.system import UniverSRSystem  # noqa: E402

logging.basicConfig(level=logging.WARNING)
SR = 48000
TCFG = dict(window_fn="hann", n_fft=1024, sampling_rate=SR, hop_length=512, alpha=0.2, beta=1, comp_eps=1e-4)
SMALL = dict(in_channels=2, out_channels=2, dims=[16, 32, 64, 128], depths=[1, 1, 2, 1], time_dim=32,
             cond_dim=32, total_freq_bins=512, hr_freq_bins=432, feature_enc_layers=2, cond_dropout_prob=0.1)


def pair(n, seed, cut_hz):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / SR
    hq = sum(0.1 * torch.sin(2 * math.pi * f * t + torch.rand(1, generator=g) * 6) for f in (300, 1200, 5000, 14000, 19000))
    k = torch.fft.rfft(hq)
    k[int(cut_hz / (SR / n)):] = 0
    lq = torch.fft.irfft(k, n=n)
    return hq[None], lq[None]


class Train(Dataset):
    def __len__(self): return 8
    def __getitem__(self, i):
        hq, lq = pair(32767, i, 16000)
        return hq, lq, torch.tensor(16000.0)


class Val(Dataset):
    def __init__(self):
        self.names = [f"song{s}_{c:04d}" for s in "ABC" for c in (0, 1)]
        self.pairs = [(f"/x/LQ/{n}.wav", f"/x/HQ/{n}.wav") for n in self.names]
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        hq, lq = pair(SR, 100 + i, 16000)
        return hq, lq, i, self.names[i], torch.tensor(16000.0)


class DM(pl.LightningDataModule):
    def setup(self, stage=None):
        self.data_train, self.data_val = Train(), Val()
    def train_dataloader(self): return DataLoader(self.data_train, batch_size=2, shuffle=True)
    def val_dataloader(self): return DataLoader(self.data_val, batch_size=1)


def make_system(tmp, **kw):
    torch.manual_seed(0)
    model = build_model(SMALL, aligned_input=kw.pop("aligned", False))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return UniverSRSystem(model, make_transform(TCFG), optimizer=opt, sample_rate=SR,
                          val_ode_steps=1, val_guidance=1.5, val_audio_dir=os.path.join(tmp, "audio"), **kw)


def trainer(tmp, steps, **kw):
    return pl.Trainer(max_steps=steps, accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
                      enable_model_summary=False, default_root_dir=tmp, val_check_interval=2, check_val_every_n_epoch=None,
                      num_sanity_val_steps=1, limit_val_batches=1.0, **kw)


def test_fit_validate_rotate_resume(aligned=False):
    with tempfile.TemporaryDirectory() as tmp:
        sysm = make_system(tmp, aligned=aligned, ema_decay=0.9, val_songs=1, val_rotate_every=2)
        tr = trainer(tmp, 6)
        w0 = sysm.audio_model.final_conv.weight.detach().clone()
        tr.fit(sysm, datamodule=DM())
        assert tr.global_step == 6
        assert not torch.equal(w0, sysm.audio_model.final_conv.weight), "weights did not update"
        m = tr.callback_metrics
        for k in ("train_loss", "sisdr", "hfnr", "visqol", "lsd_high", "val_cfm"):
            assert k in m, f"missing metric {k}"
        assert math.isfinite(float(m["train_loss"])) and math.isfinite(float(m["sisdr"]))
        assert sysm._val_window_idx >= 1, "val window never rotated"
        out = os.path.join(tmp, "audio")
        steps = sorted(os.listdir(out))
        files = os.listdir(os.path.join(out, steps[-1]))
        assert len(files) == 6, files
        ck = os.path.join(tmp, "last.ckpt")
        tr.save_checkpoint(ck)
        blob = torch.load(ck, weights_only=False)
        assert "ema" in blob and blob["val_song_keys"] == ["songA", "songB", "songC"]
        idx, step = sysm._val_window_idx, tr.global_step

        sys2 = make_system(tmp, aligned=aligned, ema_decay=0.9, val_songs=1, val_rotate_every=2)
        tr2 = trainer(tmp, 8)
        tr2.fit(sys2, datamodule=DM(), ckpt_path=ck)
        assert tr2.global_step == 8 and step == 6
        assert sys2._val_window_idx >= idx, "window position lost on resume"
        assert sys2._ema is not None
        print(f"  fit/validate/rotate/resume ok (aligned={aligned}): window idx {idx} -> {sys2._val_window_idx}, "
              f"sisdr {float(m['sisdr']):.1f}, cfm {float(m['val_cfm']):.3f}")


def test_validate_only_baseline():
    with tempfile.TemporaryDirectory() as tmp:
        sysm = make_system(tmp, val_songs=2)
        res = trainer(tmp, 1).validate(sysm, datamodule=DM(), verbose=False)
        assert res and "sisdr" in res[0] and "visqol" in res[0]
        print("  trainer.validate baseline works:", {k: round(v, 2) for k, v in res[0].items() if k != "lr"})


if __name__ == "__main__":
    test_fit_validate_rotate_resume(False)
    test_fit_validate_rotate_resume(True)
    test_validate_only_baseline()
    print("all passed")
