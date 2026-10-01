import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

from universr.models.loader import build_model, load_state_dict_into  # noqa: E402
from universr.models.unet import ConvNeXtUNetCond  # noqa: E402

logging.basicConfig(level=logging.WARNING)

SR_TO_BINS = {8: 80, 12: 128, 16: 170, 24: 256}
SMALL = dict(in_channels=2, out_channels=2, dims=[16, 32, 64, 128], depths=[1, 1, 2, 1],
             drop_path=0., time_dim=32, cond_dim=32, total_freq_bins=512, hr_freq_bins=432,
             feature_enc_layers=2, cond_dropout_prob=0.1, sr_to_lr_bins=SR_TO_BINS)
T = 128


def make_pair(**kw):
    torch.manual_seed(0)
    orig = ConvNeXtUNetCond(**SMALL).eval()
    for p in orig.parameters():
        p.data.add_(torch.randn_like(p) * 0.05)
    mod = build_model(SMALL, **kw).eval()
    load_state_dict_into(mod, orig.state_dict())
    return orig, mod


def test_step0_equivalence(**kw):
    orig, mod = make_pair(**kw)
    for sr, bins in SR_TO_BINS.items():
        torch.manual_seed(sr)
        x = torch.randn(2, 2, 432, T)
        y_full = torch.randn(2, 2, 512, T)
        t = torch.rand(2)
        with torch.no_grad():
            ref = orig(x, t, y_full[:, :, :bins], sr)
            out = mod(x, t, y_full if kw.get("aligned_input") else y_full[:, :, :bins],
                      cutoff_bins=torch.tensor([bins, bins]))
            ref_u = orig(x, t, None, sr)
            out_u = mod(x, t, None, cutoff_bins=torch.tensor([bins, bins]))
        d = (ref - out).abs().max().item()
        du = (ref_u - out_u).abs().max().item()
        assert d < 1e-4 and du < 1e-4, f"sr={sr} cond diff {d:.2e} uncond diff {du:.2e}"
        print(f"  sr={sr:>2} bins={bins:>3}  cond {d:.2e}  uncond {du:.2e}")


def test_hetero_batch_and_grads():
    _, mod = make_pair(aligned_input=True)
    mod.train()
    x = torch.randn(3, 2, 432, T)
    y = torch.randn(3, 2, 512, T)
    out = mod(x, torch.rand(3), y, cutoff_bins=torch.tensor([80, 341, 470]))
    assert out.shape == x.shape and torch.isfinite(out).all()
    out.square().mean().backward()
    assert mod.init_conv[0].weight.grad[:, 2 + 32:].abs().sum() > 0
    assert mod.bw_embedder.weight.grad.abs().sum() > 0
    print("  hetero batch ok, grads reach aligned channels and bandwidth table")


def test_gen_start_zero():
    torch.manual_seed(0)
    cfg = {k: v for k, v in SMALL.items() if k != "hr_freq_bins"}
    mod = build_model(cfg, gen_start_bin=0, aligned_input=True).eval()
    x = torch.randn(2, 2, 512, T)
    y = torch.randn(2, 2, 512, T)
    out = mod(x, torch.rand(2), y, cutoff_bins=[341, 341])
    assert out.shape == x.shape
    print("  gen_start=0 ok:", tuple(out.shape))


def test_checkpoint_matches():
    _, a = make_pair(aligned_input=True)
    _, b = make_pair(aligned_input=True, grad_checkpoint=True)
    b.load_state_dict(a.state_dict())
    a.train(); b.train()
    for m in (a, b):
        m.cond_dropout_prob = 0.0
    x, y, t = torch.randn(2, 2, 432, T), torch.randn(2, 2, 512, T), torch.rand(2)
    cb = torch.tensor([128, 341])
    oa = a(x, t, y, cutoff_bins=cb)
    ob = b(x, t, y, cutoff_bins=cb)
    oa.sum().backward(); ob.sum().backward()
    assert (oa - ob).abs().max() < 1e-5
    ga = a.init_conv[0].weight.grad
    gb = b.init_conv[0].weight.grad
    assert (ga - gb).abs().max() < 1e-4
    print("  checkpointing matches outputs and grads")


if __name__ == "__main__":
    print("equivalence, arm B layout"); test_step0_equivalence()
    print("equivalence, arm C layout (zero-init aligned channels)"); test_step0_equivalence(aligned_input=True)
    test_hetero_batch_and_grads()
    test_gen_start_zero()
    test_checkpoint_matches()
    print("all passed")
