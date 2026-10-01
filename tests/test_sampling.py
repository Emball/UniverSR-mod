import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

from universr.models.loader import build_model  # noqa: E402
from universr.sampling import (assemble, make_transform, ode_sample, restore,  # noqa: E402
                               restore_long, to_spec, to_wave)

TCFG = dict(window_fn="hann", n_fft=1024, sampling_rate=48000, hop_length=512,
            alpha=0.2, beta=1, comp_eps=1e-4)
SMALL = dict(in_channels=2, out_channels=2, dims=[16, 32, 64, 128], depths=[1, 1, 2, 1],
             time_dim=32, cond_dim=32, total_freq_bins=512, hr_freq_bins=432, feature_enc_layers=2,
             cond_dropout_prob=0.0)
SR = 48000


def test_stft_roundtrip():
    tr = make_transform(TCFG)
    torch.manual_seed(0)
    x = torch.randn(2, 1, 32767) * 0.2
    y = to_wave(tr, to_spec(tr, x, pad_to_hop=True), x.shape[-1])
    assert y.shape == x.shape
    rms = (x - y).square().mean().sqrt().item()
    peak = (x - y).abs().max().item()
    print(f"  roundtrip (Nyquist dropped, padded): rms err {rms:.2e}, max err {peak:.2e}")
    assert peak < 0.05
    raw = to_spec(tr, x)
    assert raw.shape[-1] == 64, "training framing must match upstream (32767 samples -> 64 frames)"
    print("  unpadded framing matches upstream: 32767 samples -> 64 frames")


def test_tail_stable_under_modified_spectrum():
    tr = make_transform(TCFG)
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32767) * 0.2
    spec = to_spec(tr, x, pad_to_hop=True)
    mod = spec * (1 + 0.3 * torch.randn_like(spec))
    y = to_wave(tr, mod, x.shape[-1])
    body, tail = y[..., :-600].abs().max().item(), y[..., -600:].abs().max().item()
    print(f"  perturbed spectrum: body peak {body:.2f}, last-600-sample peak {tail:.2f}")
    assert tail < 3 * body


class Linear:
    hr_freq_bins = 4
    def __call__(self, x, t, y, cb):
        return torch.ones_like(x) * 2.0


def test_midpoint_exact_on_constant_field():
    x0_seed = 7
    out = ode_sample(Linear(), torch.zeros(1, 2, 8, 5), torch.tensor([4.0]), 5, steps=4, guidance=0, seed=x0_seed)
    g = torch.Generator().manual_seed(x0_seed)
    x0 = torch.randn(1, 2, 4, 5, generator=g)
    assert torch.allclose(out, x0 + 2.0, atol=1e-6)
    print("  midpoint solver exact on constant field; seeded noise reproducible")


def test_assemble_semantics():
    Y = torch.randn(2, 2, 512, 6)
    x1 = torch.randn(2, 2, 432, 6)
    cb = torch.tensor([128.0, 341.0])
    b = assemble(Y, x1, cb, 80, True)
    c = assemble(Y, x1, cb, 80, False)
    assert torch.equal(b[0, :, :128], Y[0, :, :128]) and torch.equal(b[0, :, 128:], x1[0, :, 128 - 80:])
    assert torch.equal(b[1, :, :341], Y[1, :, :341]) and torch.equal(b[1, :, 341:], x1[1, :, 341 - 80:])
    assert torch.equal(c[:, :, :80], Y[:, :, :80]) and torch.equal(c[:, :, 80:], x1)
    print("  assemble: arm B keeps LQ below each sample's cutoff, arm C takes generated bins from region start")


def test_restore_and_ola():
    tr = make_transform(TCFG)
    torch.manual_seed(0)
    m = build_model(SMALL).eval()
    n = SR * 2
    t = torch.arange(n) / SR
    lq = (0.3 * torch.sin(2 * math.pi * 440 * t))[None, None]
    out = restore(m, tr, lq, 16000.0, SR, steps=2, guidance=1.5, seed=1)
    assert out.shape == lq.shape and torch.isfinite(out).all()
    again = restore(m, tr, lq, 16000.0, SR, steps=2, guidance=1.5, seed=1)
    assert torch.equal(out, again)
    ola = restore_long(m, tr, lq, 16000.0, SR, chunk_sec=1.0, overlap_sec=0.5, steps=2, guidance=1.5, seed=1)
    assert ola.shape == lq.shape and torch.isfinite(ola).all()
    mc = build_model(SMALL, aligned_input=True).eval()
    oc = restore(mc, tr, lq, torch.tensor([16000.0]), SR, steps=2, guidance=0, seed=1)
    assert oc.shape == lq.shape
    print("  restore: shapes ok, seeded runs identical, OLA runs, arm C path runs")


if __name__ == "__main__":
    test_stft_roundtrip()
    test_tail_stable_under_modified_spectrum()
    test_midpoint_exact_on_constant_field()
    test_assemble_semantics()
    test_restore_and_ola()
    print("all passed")
