import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy import signal as ss

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "core"))

from ckpt_utils import best_checkpoint, list_checkpoints, parse_ckpt_name  # noqa: E402
from universr.models.unet import ConvNeXtUNetCond  # noqa: E402

NAME = "smoke_e2e"
SR_SRC = 44100
SMALL = dict(in_channels=2, out_channels=2, dims=[16, 32, 64, 128], depths=[1, 1, 2, 1], time_dim=32, cond_dim=32,
             total_freq_bins=512, hr_freq_bins=432, feature_enc_layers=2, cond_dropout_prob=0.1)
TFM = dict(window_fn="hann", n_fft=1024, sampling_rate=48000, hop_length=512, alpha=0.2, beta=1, comp_eps=1e-4)


def make_song(seconds, seed, cutoff=16000):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR_SRC)) / SR_SRC
    x = sum(a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f, a in
            [(220, .3), (440, .2), (1760, .12), (5000, .08), (9000, .06), (14000, .05), (18000, .04)])
    x = x * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 0.7 * t))) + rng.normal(0, 0.003, t.size)
    x = (x / np.abs(x).max() * 0.6).astype(np.float32)
    b, a = ss.butter(8, cutoff / (SR_SRC / 2))
    return x, ss.filtfilt(b, a, x).astype(np.float32)


def build_data():
    base = os.path.join(ROOT, "data", NAME)
    for split, n, sec in (("train", 3, 9), ("val", 2, 30)):
        for side in ("LQ", "HQ"):
            os.makedirs(os.path.join(base, split, side), exist_ok=True)
        for i in range(n):
            hq, lq = make_song(sec, 100 * (split == "val") + i)
            sf.write(os.path.join(base, split, "HQ", f"song{i}.wav"), np.stack([hq, hq], 1), SR_SRC)
            sf.write(os.path.join(base, split, "LQ", f"song{i}.wav"), np.stack([lq, lq], 1), SR_SRC)


def write_cfg(tmp, max_steps, weights):
    cfg = {
        "exp": {"dir": os.path.join(ROOT, "runs"), "name": NAME}, "seed": 1, "weights_path": weights,
        "optimizations": {"ram_limit_fraction": 0.99, "triton_cache": False},
        "training": {"max_steps": max_steps, "grad_accum_steps": 2, "grad_clip": 1.0, "precision": "auto",
                     "ema_decay": 0.9, "val_songs": 1, "val_rotate_every": "auto", "val_check_interval": 3,
                     "freeze": ["time_embedder"]},
        "datas": {"sr": 48000, "segment_sec": 3, "crop_samples": 32767, "batch_size": 2, "num_workers": 0,
                  "pin_memory": False, "cutoff_hz": "auto", "val_bootstrap_chunks": 4,
                  "augmentation": {"live": {"enabled": False}, "cached": {"enabled": False}}},
        "transform": TFM,
        "model": dict(SMALL, aligned_input=False, grad_checkpoint=True),
        "optimizer": {"type": "adamw", "lr": 1e-3, "weight_decay": 0.01, "betas": [0.9, 0.99]},
        "scheduler": {"type": "cosine", "warmup_steps": 2, "min_lr_ratio": 0.1},
        "system": {"val_ode_steps": 1, "val_guidance": 1.5, "visqol_fraction": 1.0},
    }
    path = os.path.join(tmp, f"{NAME}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def run_train(cfg_path, *extra, timeout=900):
    return subprocess.run([sys.executable, os.path.join(ROOT, "core", "train.py"), "--conf_dir", cfg_path, *extra],
                          cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def latest_run():
    base = os.path.join(ROOT, "runs", NAME)
    return os.path.join(base, sorted(os.listdir(base))[-1])


def steps_of(run):
    return sorted(int(i["step"]) for i in list_checkpoints(os.path.join(run, "checkpoints")))


def clean():
    shutil.rmtree(os.path.join(ROOT, "data", NAME), ignore_errors=True)
    shutil.rmtree(os.path.join(ROOT, "runs", NAME), ignore_errors=True)


def main():
    clean()
    tmp = tempfile.mkdtemp()
    try:
        torch.manual_seed(0)
        weights = os.path.join(tmp, "pretrained.bin")
        torch.save(ConvNeXtUNetCond(**dict(SMALL, drop_path=0., sr_to_lr_bins={8: 80, 12: 128, 16: 170, 24: 256})).state_dict(), weights)
        build_data()

        r = run_train(write_cfg(tmp, 6, weights))
        print(r.stdout[-1800:] if r.returncode else "", r.stderr[-1500:] if r.returncode else "")
        assert r.returncode == 0, "fresh run failed"
        out = r.stdout
        assert "[baseline]" in out and "Training finished!" in out
        run = latest_run()
        names = os.listdir(os.path.join(run, "checkpoints"))
        assert all(parse_ckpt_name(n) for n in names), names
        assert steps_of(run) == [3, 6], steps_of(run)
        assert any("visqol=" in n and "sisdr=" in n and "hfnr=" in n for n in names)
        assert os.path.isfile(os.path.join(run, "best_model.pth")) and os.path.isfile(os.path.join(run, "best_k_models.json"))
        bundle = torch.load(os.path.join(run, "best_model.pth"), weights_only=False)
        assert bundle["gen_start_bin"] == 80 and bundle["aligned_input"] is False and "model_state_dict" in bundle
        print(f"  fresh run ok: checkpoints {sorted(names)}")
        print(f"  best: {os.path.basename(best_checkpoint(os.path.join(run, 'checkpoints')))}")

        cfg2 = write_cfg(tmp, 9, weights)
        r2 = run_train(cfg2, "--resume")
        assert r2.returncode == 0, r2.stdout[-1500:] + r2.stderr[-1500:]
        assert "[resume] run" in r2.stdout and "[baseline]" not in r2.stdout.split("[resume]")[-1]
        assert steps_of(run) == [3, 6, 9], steps_of(run)
        assert len(os.listdir(os.path.join(ROOT, "runs", NAME))) == 1, "resume created a second run"
        print("  resume ok: steps 3,6 -> 9 in the same run folder, no baseline re-run")

        cfg3 = write_cfg(tmp, 100000, weights)
        p = subprocess.Popen([sys.executable, os.path.join(ROOT, "core", "train.py"), "--conf_dir", cfg3, "--resume"],
                             cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        seen, deadline = "", time.time() + 600
        while time.time() < deadline:
            line = p.stdout.readline()
            seen += line
            if re.search(r"step=1[0-9]\b|step=[2-9]\d\b", line) and "it/s" in line:
                p.send_signal(signal.SIGINT)
                break
            if not line and p.poll() is not None:
                break
        rest = p.communicate(timeout=300)[0]
        seen += rest
        assert p.returncode == 0, seen[-1500:]
        assert "[interrupt] saved" in seen, seen[-1500:]
        saved = [n for n in os.listdir(os.path.join(run, "checkpoints")) if re.fullmatch(r"step=\d+\.ckpt", n)]
        assert saved, os.listdir(os.path.join(run, "checkpoints"))
        assert best_checkpoint(os.path.join(run, "checkpoints")) and "visqol" in os.path.basename(best_checkpoint(os.path.join(run, "checkpoints")))
        print(f"  Ctrl+C ok: saved {saved}; interrupted checkpoint is never ranked best")
        print("all passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        clean()


if __name__ == "__main__":
    main()
