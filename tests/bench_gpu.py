"""GPU speed/memory benchmark and fp16 NaN locator for the U-Net. Run on the GPU machine:

    python tests/bench_gpu.py --conf_dir configs/universr_stfl_new.yaml
    python tests/bench_gpu.py --conf_dir configs/universr_stfl_new.yaml --batches 8 --skip_nan

Prints ms per training step and peak VRAM for several model variants, then runs the pretrained weights in
fp16 and reports the first module that produces a non-finite value.
"""
import argparse
import math
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "core"))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from universr.models.loader import build_model, load_pretrained  # noqa: E402

VARIANTS = [
    ("original ops, ckpt on", dict(fast_ops=False, channels_last=False, ck=True)),
    ("fast ops, ckpt on", dict(fast_ops=True, channels_last=False, ck=True)),
    ("fast ops + channels_last, ckpt on", dict(fast_ops=True, channels_last=True, ck=True)),
    ("fast ops + channels_last, ckpt off", dict(fast_ops=True, channels_last=True, ck=False)),
]


def make_model(mc, v):
    m = dict(mc)
    m["fast_ops"], m["channels_last"] = v["fast_ops"], v["channels_last"]
    return build_model(m, gen_start_bin=m.get("gen_start_bin"), aligned_input=bool(m.get("aligned_input", False)),
                       bw_anchor_bins=m.get("bw_anchor_bins"), grad_checkpoint=v["ck"])


def make_inputs(model, B, T, dev):
    Fg, total = model.hr_freq_bins, model.total_freq_bins
    cut = 341
    x = torch.randn(B, 2, Fg, T, device=dev) * 0.5
    y = torch.randn(B, 2, cut, T, device=dev) * 0.5
    t = torch.rand(B, 1, 1, 1, device=dev)
    cb = torch.full((B,), float(cut), device=dev)
    return x, t, y, cb


def bench(mc, v, B, T, steps, dev):
    torch.cuda.empty_cache()
    model = make_model(mc, v).to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    x, t, y, cb = make_inputs(model, B, T, dev)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(steps + 3):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x, t, y, cb)
            loss = out.float().square().mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        if i >= 3:
            times.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model, opt, x, y, out, loss
    return sum(times) / len(times) * 1000, peak


def locate_nonfinite(model, B, T, dev, label, autocast):
    model.eval()
    x, t, y, cb = make_inputs(model, B, T, dev)
    first, peaks, hooks = [None], [], []

    def mk(name):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) and out else out
            if not torch.is_tensor(o) or not o.is_floating_point():
                return
            mx = o.detach().abs().max().item()
            peaks.append((mx if math.isfinite(mx) else float("inf"), name, str(o.dtype)))
            if first[0] is None and not math.isfinite(mx):
                first[0] = (name, str(o.dtype))
        return hook

    for name, m in model.named_modules():
        if name:
            hooks.append(m.register_forward_hook(mk(name)))
    with torch.no_grad():
        if autocast:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x, t, y, cb)
        else:
            out = model(x, t, y, cb)
    for h in hooks:
        h.remove()
    finite = bool(torch.isfinite(out).all())
    print(f"  {label}: output finite={finite}; first non-finite module: {first[0]}")
    top = sorted((p for p in peaks if math.isfinite(p[0])), reverse=True)[:3]
    print("    largest finite activations: " + ", ".join(f"{n} ({d}) {v:.0f}" for v, n, d in top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf_dir", default="configs/universr_stfl_new.yaml")
    ap.add_argument("--weights", default=None, help="pretrained .bin (default: models/pytorch_model.bin)")
    ap.add_argument("--batches", default="4,8", help="comma separated batch sizes")
    ap.add_argument("--frames", type=int, default=64, help="time frames per sample (64 = the 32767-sample crop)")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--skip_nan", action="store_true")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 2**30
    print(f"{props.name}  {total:.1f} GB  capability {props.major}.{props.minor}  torch {torch.__version__}")

    cfg = OmegaConf.load(os.path.join(REPO, a.conf_dir) if not os.path.isabs(a.conf_dir) else a.conf_dir)
    mc = OmegaConf.to_container(cfg.model, resolve=True)

    print(f"\ntraining step (fp16 autocast, {a.frames} frames per sample, includes optimizer step)")
    for B in [int(b) for b in a.batches.split(",")]:
        for label, v in VARIANTS:
            try:
                ms, peak = bench(mc, v, B, a.frames, a.steps, dev)
                warn = "  <-- near the VRAM limit, Windows may spill to system RAM" if peak > 0.9 * total else ""
                print(f"  batch {B:2d}  {label:38s} {ms:8.0f} ms/step  peak {peak:5.2f} GB{warn}")
            except torch.cuda.OutOfMemoryError:
                print(f"  batch {B:2d}  {label:38s} OOM")
                torch.cuda.empty_cache()

    if a.skip_nan:
        return
    path = a.weights
    if path is None:
        cand = os.path.join(REPO, "models", "pytorch_model.bin")
        path = cand if os.path.isfile(cand) else None
    if path is None:
        print("\nno pretrained weights found, skipping the fp16 check")
        return
    print(f"\nfp16 check with pretrained weights ({os.path.basename(path)}), random spectra as input")
    model = make_model(mc, dict(fast_ops=True, channels_last=False, ck=False))
    load_pretrained(model, path)
    model.to(dev)
    for B, T, tag in ((1, 937, "10 s clip"), (8, a.frames, "training batch")):
        for ac, name in ((False, "fp32"), (True, "fp16 autocast")):
            try:
                locate_nonfinite(model, B, T, dev, f"{tag}, {name}", ac)
            except torch.cuda.OutOfMemoryError:
                print(f"  {tag}, {name}: OOM")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
