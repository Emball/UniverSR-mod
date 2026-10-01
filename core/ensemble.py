import json
import logging

import torch

from universr.sampling import ola_window

log = logging.getLogger("universr.ensemble")

MODES = ("max_fft", "min_fft", "avg", "original", "enhanced")
N_FFT = 4096


def parse_bands(spec):
    if spec is None or spec == "":
        return None
    if isinstance(spec, str):
        spec = json.loads(spec)
    bands = []
    for b in spec:
        mode = str(b.get("mode", "enhanced")).lower()
        if mode not in MODES:
            raise ValueError(f"unknown ensemble mode {mode!r}; use one of {MODES}")
        bands.append({
            "lo": float(b.get("lo", 0.0)),
            "hi": float(b["hi"]) if b.get("hi") is not None else None,
            "mode": mode,
            "weight": max(0.0, min(1.0, float(b.get("weight", 1.0)))),
        })
    return bands


def low_end_bands(hz=700.0):
    return [{"lo": 0.0, "hi": float(hz), "mode": "max_fft", "weight": 1.0}]


def spectral_merge(original, enhanced, sr, bands):
    """Blend magnitudes of original and enhanced ([C,T]) per band; phase always comes from enhanced."""
    T = enhanced.shape[-1]
    original = original[..., :T]
    pad = max(0, N_FFT - T)
    if pad:
        enhanced = torch.nn.functional.pad(enhanced, (0, pad))
        original = torch.nn.functional.pad(original, (0, pad))
    hop = N_FFT // 4
    window = torch.hann_window(N_FFT, device=enhanced.device)
    kw = dict(n_fft=N_FFT, hop_length=hop, win_length=N_FFT, window=window)
    S_o = torch.stft(original, return_complex=True, pad_mode="reflect", **kw)
    S_e = torch.stft(enhanced, return_complex=True, pad_mode="reflect", **kw)
    mag_o, mag_e = S_o.abs(), S_e.abs()
    phase = S_e / (mag_e + 1e-8)
    mag_out = mag_e.clone()

    bin_hz = sr / N_FFT
    n_bins = N_FFT // 2 + 1
    for b in bands:
        lo = max(0, int(b["lo"] / bin_hz))
        hi_hz = b["hi"] if b["hi"] is not None else sr / 2
        hi = min(n_bins, int(hi_hz / bin_hz) + 1)
        if lo >= hi:
            continue
        mo, me = mag_o[:, lo:hi], mag_e[:, lo:hi]
        if b["mode"] == "max_fft":
            m = torch.maximum(mo, me)
        elif b["mode"] == "min_fft":
            m = torch.minimum(mo, me)
        elif b["mode"] == "avg":
            m = 0.5 * (mo + me)
        elif b["mode"] == "original":
            m = mo
        else:
            m = me
        mag_out[:, lo:hi] = b["weight"] * m + (1.0 - b["weight"]) * me

    out = torch.istft(mag_out * phase, length=enhanced.shape[-1], **kw)
    return out[..., :T]


def merge_long(original, enhanced, sr, bands, block_sec=30.0, overlap_sec=1.0):
    """spectral_merge in overlapping blocks so long files stay within RAM."""
    T = enhanced.shape[-1]
    block = int(block_sec * sr)
    if T <= block:
        return spectral_merge(original, enhanced, sr, bands)
    overlap = min(int(overlap_sec * sr), block - 1)
    hop = block - overlap
    out = torch.zeros_like(enhanced)
    wsum = torch.zeros(T, device=enhanced.device)
    start = 0
    while start < T:
        end = min(start + block, T)
        piece = spectral_merge(original[..., start:end], enhanced[..., start:end], sr, bands)
        w = ola_window(end - start, overlap if start > 0 else 0, overlap if end < T else 0, enhanced.device)
        out[..., start:end] += piece * w
        wsum[start:end] += w
        if end == T:
            break
        start += hop
    return out / wsum.clamp(min=1e-8)
