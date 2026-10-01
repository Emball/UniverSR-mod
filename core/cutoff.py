import logging

import numpy as np

log = logging.getLogger(__name__)

_EPS = 1e-12


def _mono(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=0)
    return wav


def mean_power_db(wav: np.ndarray, n_fft: int = 2048, max_frames: int = 400) -> np.ndarray:
    x = _mono(wav)
    hop = n_fft // 2
    n_frames = 1 + max(0, (len(x) - n_fft) // hop)
    if n_frames < 1:
        x = np.pad(x, (0, n_fft - len(x)))
        n_frames = 1
    if n_frames > max_frames:
        starts = np.linspace(0, (n_frames - 1) * hop, max_frames).astype(int)
    else:
        starts = np.arange(n_frames) * hop
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.stack([x[s:s + n_fft] * win for s in starts])
    power = (np.abs(np.fft.rfft(frames, axis=-1)) ** 2).mean(axis=0) / (n_fft * 0.375)
    return 10.0 * np.log10(power + _EPS)


def estimate_cutoff_hz(
    wav: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    min_drop_db: float = 20.0,
    smooth_bins: int = 5,
    window_hz: float = 1000.0,
    min_hz: float = 2000.0,
    silence_db: float = -90.0,
    edge_db: float = 6.0,
    fallback_hz: float | None = None,
) -> float:
    nyq = sr / 2.0
    fb = nyq if fallback_hz is None else float(fallback_hz)
    db = mean_power_db(wav, n_fft)
    if db.max() < silence_db:
        return fb

    k = np.ones(smooth_bins, dtype=np.float32) / smooth_bins
    pad = (smooth_bins // 2, smooth_bins - 1 - smooth_bins // 2)
    sm = np.convolve(np.pad(db, pad, mode="edge"), k, mode="valid")
    n = len(sm)
    w = max(2, int(round(window_hz * n_fft / sr)))
    ext = np.concatenate([sm, np.full(w, sm[-1], dtype=sm.dtype)])
    csum = np.concatenate([[0.0], np.cumsum(ext)])
    idx = np.arange(n)
    before = (csum[idx] - csum[np.maximum(idx - w, 0)]) / np.maximum(np.minimum(idx, w), 1)
    after = (csum[idx + w] - csum[idx]) / w
    drop = before - after
    lo = int(min_hz * n_fft / sr)
    drop[:lo] = -np.inf
    i = int(np.argmax(drop))
    if drop[i] < min_drop_db:
        return nyq
    ref = float(np.mean(sm[max(lo, i - 3 * w):max(lo + 1, i - w)]))
    j = i
    while j > lo and sm[j] < ref - edge_db:
        j -= 1
    return float(np.clip(j * sr / n_fft, min_hz, nyq))


def resolve_cutoff_hz(spec, wav: np.ndarray, sr: int, rng=None) -> float:
    nyq = sr / 2.0
    if spec is None or spec == "auto":
        return estimate_cutoff_hz(wav, sr)
    if isinstance(spec, (int, float)):
        return float(min(spec, nyq))
    lo, hi = float(spec[0]), float(spec[1])
    r = rng.uniform(lo, hi) if rng is not None else np.random.uniform(lo, hi)
    return float(min(r, nyq))
