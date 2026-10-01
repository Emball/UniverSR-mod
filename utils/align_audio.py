import argparse
import sys
import os
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, correlate
from scipy.interpolate import interp1d
from math import gcd
from fractions import Fraction

def load_audio(path: str):
    """Load audio file. Returns (data: np.ndarray [samples, channels], sr: int)."""
    data, sr = sf.read(path, always_2d=True, dtype="float64")
    return data, sr


def save_audio(path: str, data: np.ndarray, sr: int):
    """Save audio file, clipping to [-1, 1]."""
    data = np.clip(data, -1.0, 1.0)
    sf.write(path, data, sr, subtype="PCM_24")
    print(f"  Saved → {path}  ({data.shape[0]} samples, {data.shape[0]/sr:.4f}s)")


def to_mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1)


def check_stereo_swap(hq: np.ndarray, lq: np.ndarray) -> np.ndarray:
    if hq.shape[1] < 2 or lq.shape[1] < 2:
        return lq
    
    check_len = min(hq.shape[0], lq.shape[0], 44100 * 10)
    start_smp = (min(hq.shape[0], lq.shape[0]) // 2) - (check_len // 2)
    
    hq_l = hq[start_smp : start_smp + check_len, 0]
    lq_l = lq[start_smp : start_smp + check_len, 0]
    lq_r = lq[start_smp : start_smp + check_len, 1]
    
    # Simple correlation at zero lag
    corr_normal = np.abs(np.vdot(hq_l, lq_l))
    corr_swapped = np.abs(np.vdot(hq_l, lq_r))
    
    if corr_swapped > corr_normal:
        print("  !!! Stereo swap detected on LQ input. Fixing channels.")
        return lq[:, [1, 0]]
    return lq


def match_rms(hq: np.ndarray, lq: np.ndarray) -> np.ndarray:
    rms_hq = np.sqrt(np.mean(np.square(hq)))
    rms_lq = np.sqrt(np.mean(np.square(lq)))
    if rms_lq < 1e-9:
        return lq
    gain = rms_hq / rms_lq
    print(f"  RMS Leveling: Scaling LQ by {gain:.4f}x to match HQ")
    return lq * gain


def prepare_proxy(data: np.ndarray) -> np.ndarray:
    x = to_mono(data)
    
    hard = np.clip(x, -1.0, 1.0)
    
    x_clamped = np.clip(x, -1.0, 1.0)
    soft = 1.5 * x_clamped - 0.5 * (x_clamped ** 3)
   
    proxy = 0.5 * hard + 0.5 * soft
    
    peak = np.max(np.abs(proxy))
    if peak > 1e-9:
        proxy = proxy / peak
        
    return proxy


def sinc_resample(data: np.ndarray, src_len: int, dst_len: int) -> np.ndarray:
    if src_len == dst_len:
        return data.copy()

    ratio = Fraction(dst_len, src_len).limit_denominator(8000)
    up, down = ratio.numerator, ratio.denominator

    # Reduce by GCD to keep resample_poly happy
    g = gcd(up, down)
    up, down = up // g, down // g

    channels = data.shape[1] if data.ndim == 2 else 1
    if channels == 1:
        resampled = resample_poly(data.ravel(), up, down)
        if data.ndim == 2:
            resampled = resampled[:, np.newaxis]
    else:
        parts = [resample_poly(data[:, c], up, down) for c in range(channels)]
        resampled = np.stack(parts, axis=1)

    # Trim or pad to exact dst_len (rational approx may be off by a few samples)
    current_len = resampled.shape[0]
    if current_len > dst_len:
        resampled = resampled[:dst_len]
    elif current_len < dst_len:
        pad = dst_len - current_len
        if resampled.ndim == 2:
            resampled = np.pad(resampled, ((0, pad), (0, 0)))
        else:
            resampled = np.pad(resampled, (0, pad))

    return resampled

def measure_offset_xcorr(
    hq_window: np.ndarray,
    lq_window: np.ndarray,
    max_shift_samples: int,
) -> float:
    corr = correlate(hq_window, lq_window, mode="full")
    mid = len(corr) // 2
    lo = max(0, mid - max_shift_samples)
    hi = min(len(corr), mid + max_shift_samples + 1)
    local = corr[lo:hi]
    peak_local = int(np.argmax(np.abs(local)))
    peak_global = peak_local + lo

    if 0 < peak_global < len(corr) - 1:
        y0, y1, y2 = corr[peak_global - 1], corr[peak_global], corr[peak_global + 1]
        denom = 2 * (2 * y1 - y0 - y2)
        frac = (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        frac = 0.0

    offset = (peak_global + frac) - mid
    return offset


def find_global_ratio(
    hq_audio: np.ndarray,
    lq_audio: np.ndarray,
    anchor_ms: float = 500.0,
    sr: int = 44100,
    search_ppm: int = 3000,
    search_steps: int = 600,  # kept for API compatibility, not used
) -> float:
    hq_mono = prepare_proxy(hq_audio)
    lq_mono = prepare_proxy(lq_audio)

    lq_len = len(lq_mono)
    hq_len = len(hq_mono)

    # Use a ~4410Hz proxy (1/10th) — fast but much more precise than 1/20th
    PROXY_FACTOR = 10
    PROXY_SR = sr // PROXY_FACTOR

    hq_proxy = hq_mono[::PROXY_FACTOR].copy()
    lq_proxy = lq_mono[::PROXY_FACTOR].copy()

    anchor_proxy = max(128, int(anchor_ms * PROXY_SR / 1000))
    max_shift_proxy = int(search_ppm / 1_000_000 * len(lq_proxy)) + anchor_proxy

    print(f"  Proxy: {PROXY_SR}Hz  ({len(lq_proxy)} samples, anchor={anchor_ms:.0f}ms)")

    hq_start_win = hq_proxy[:anchor_proxy]
    lq_start_win = lq_proxy[:anchor_proxy]
    start_offset_proxy = measure_offset_xcorr(hq_start_win, lq_start_win, anchor_proxy)
    start_offset_samples = start_offset_proxy * PROXY_FACTOR
    print(f"  Start offset: {start_offset_proxy:+.3f} proxy samples  "
          f"({start_offset_samples:+.1f} real samples)")

    hq_end_win = hq_proxy[-anchor_proxy:]
    lq_end_win = lq_proxy[-anchor_proxy:]
    end_offset_proxy = measure_offset_xcorr(hq_end_win, lq_end_win, anchor_proxy)
    end_offset_samples_coarse = end_offset_proxy * PROXY_FACTOR
    print(f"  End offset (coarse): {end_offset_proxy:+.3f} proxy samples  "
          f"({end_offset_samples_coarse:+.1f} real samples)")

    REFINE_FACTOR = 2
    REFINE_SR = sr // REFINE_FACTOR
    hq_ref = hq_mono[::REFINE_FACTOR].copy()
    lq_ref = lq_mono[::REFINE_FACTOR].copy()
    anchor_ref = max(512, int(anchor_ms * REFINE_SR / 1000))
    # Search window for refine: ±2x the coarse estimate (in refine samples)
    coarse_in_refine = int(abs(end_offset_samples_coarse) / REFINE_FACTOR) + 64

    hq_end_ref = hq_ref[-anchor_ref:]
    lq_end_ref = lq_ref[-anchor_ref:]
    end_offset_refine = measure_offset_xcorr(hq_end_ref, lq_end_ref, coarse_in_refine + 64)
    end_offset_samples = end_offset_refine * REFINE_FACTOR
    print(f"  End offset (refined): {end_offset_refine:+.4f} refine samples  "
          f"({end_offset_samples:+.3f} real samples)")

    target_len = hq_len + end_offset_samples
    ratio = target_len / lq_len

    max_ratio = 1.0 + search_ppm / 1_000_000
    min_ratio = 1.0 - search_ppm / 1_000_000
    ratio_clamped = max(min_ratio, min(max_ratio, ratio))
    if abs(ratio_clamped - ratio) > 1e-9:
        print(f"  WARNING: computed ratio {ratio:.8f} outside ±{search_ppm}ppm — clamped to {ratio_clamped:.8f}")
        ratio = ratio_clamped

    implied_ppm = (ratio - 1.0) * 1_000_000
    print(f"  Computed ratio: {ratio:.8f}  ({implied_ppm:+.1f} ppm)")
    return ratio

def hann_fade(length: int) -> np.ndarray:
    return np.sin(np.linspace(0, np.pi / 2, length)) ** 2


def parabolic_peak(corr: np.ndarray, peak_idx: int) -> float:
    if peak_idx <= 0 or peak_idx >= len(corr) - 1:
        return float(peak_idx)
    y0, y1, y2 = corr[peak_idx - 1], corr[peak_idx], corr[peak_idx + 1]
    denom = 2 * (2 * y1 - y0 - y2)
    if abs(denom) < 1e-12:
        return float(peak_idx)
    return peak_idx + (y0 - y2) / denom


def xcorr_find_offset(
    hq_mono: np.ndarray,
    lq_mono: np.ndarray,
    lq_cursor: int,
    chunk_start: int,
    chunk_len: int,
    max_shift: int,
    n_samples: int,
) -> tuple[int, float, float]:
    hq_chunk = hq_mono[chunk_start : chunk_start + chunk_len]

    search_start = max(0, lq_cursor - max_shift)
    search_end   = min(n_samples, lq_cursor + chunk_len + max_shift)
    lq_search    = lq_mono[search_start:search_end]

    if len(lq_search) < chunk_len:
        return lq_cursor, 0.0, 0.0

    corr = correlate(lq_search, hq_chunk, mode="valid")
    corr_abs = np.abs(corr)

    expected_pos = lq_cursor - search_start
    lo = max(0, expected_pos - max_shift)
    hi = min(len(corr_abs), expected_pos + max_shift + 1)
    local_corr = corr_abs[lo:hi]

    if len(local_corr) == 0:
        return lq_cursor, 0.0, 0.0

    best_local = int(np.argmax(local_corr))
    best_pos   = best_local + lo

    mean_val = float(np.mean(local_corr)) + 1e-12
    peak_val = float(local_corr[best_local])
    confidence = min(1.0, (peak_val / mean_val - 1.0) / 10.0)

    sub = parabolic_peak(corr_abs, best_pos)
    sub_offset = sub - best_pos

    lq_chunk_start = search_start + best_pos
    lq_chunk_start = max(0, min(lq_chunk_start, n_samples - chunk_len))

    return lq_chunk_start, sub_offset, confidence


def micro_align_chunks(
    hq: np.ndarray,
    lq: np.ndarray,
    sr: int,
    chunk_ms: float = 150.0,
    crossfade_ms: float = 8.0,
    max_drift_ms: float = 50.0,
    conf_threshold: float = 0.05,
) -> np.ndarray:
    n_samples  = hq.shape[0]
    n_channels = hq.shape[1]
    chunk_len  = int(chunk_ms * sr / 1000)
    cf_len     = int(crossfade_ms * sr / 1000)
    max_shift  = int(max_drift_ms * sr / 1000)

    cf_len = min(cf_len, chunk_len // 4)
    hop = max(chunk_len - cf_len, chunk_len // 2)

    # Use internal proxies for correlation
    print("  Creating internal alignment proxies (50/50 Hard/Soft Blend)...")
    hq_mono = prepare_proxy(hq)
    lq_mono = prepare_proxy(lq)

    chunk_starts = list(range(0, n_samples, hop))
    n_chunks = len(chunk_starts)

    print(f"  Micro-aligning {n_chunks} chunks  "
          f"({chunk_ms:.0f}ms chunks, {hop/sr*1000:.1f}ms hop, "
          f"{crossfade_ms:.1f}ms crossfade, ±{max_drift_ms:.0f}ms drift, "
          f"conf_threshold={conf_threshold:.2f})")

    lq_starts   = np.zeros(n_chunks, dtype=np.int64)
    sub_offs    = np.zeros(n_chunks, dtype=np.float64)
    confidences = np.zeros(n_chunks, dtype=np.float64)

    lq_cursor = 0
    for i, chunk_start in enumerate(chunk_starts):
        c_len = min(chunk_len, n_samples - chunk_start)
        if c_len < max(8, cf_len * 2):
            lq_starts[i]   = lq_cursor
            sub_offs[i]    = 0.0
            confidences[i] = 1.0
            break

        lq_pos, sub_off, conf = xcorr_find_offset(
            hq_mono, lq_mono,
            lq_cursor, chunk_start, c_len,
            max_shift, n_samples,
        )
        lq_starts[i]   = lq_pos
        sub_offs[i]    = sub_off
        confidences[i] = conf

        if conf >= conf_threshold:
            lq_cursor = lq_pos + hop
        else:
            lq_cursor = lq_cursor + hop
        lq_cursor = max(0, min(lq_cursor, n_samples))

    high_conf_idx = np.where(confidences[:n_chunks] >= conf_threshold)[0]
    n_low = n_chunks - len(high_conf_idx)
    low_mask = confidences[:n_chunks] < conf_threshold

    if len(high_conf_idx) >= 2:
        drifts = lq_starts - np.array(chunk_starts[:n_chunks], dtype=np.int64)
        interp_fn = interp1d(
            high_conf_idx,
            drifts[high_conf_idx],
            kind="linear",
            bounds_error=False,
            fill_value=(drifts[high_conf_idx[0]], drifts[high_conf_idx[-1]]),
        )
        if np.any(low_mask):
            all_idx = np.arange(n_chunks)
            interp_drifts = interp_fn(all_idx[low_mask])
            lq_starts[low_mask] = np.round(
                np.array(chunk_starts[:n_chunks], dtype=np.float64)[low_mask] + interp_drifts
            ).astype(np.int64)
            lq_starts = np.clip(lq_starts, 0, n_samples - 1)

    print(f"  Confidence: {len(high_conf_idx)}/{n_chunks} chunks trusted  "
          f"({n_low} interpolated)")

    out    = np.zeros_like(hq)
    weight = np.zeros(n_samples)

    def extract_chunk(i, chunk_start, c_len):
        lq_chunk_start = int(lq_starts[i])
        sub_off        = sub_offs[i]
        grab_extra     = 2
        lq_grab_start  = max(0, lq_chunk_start)
        lq_grab_end    = min(n_samples, lq_chunk_start + c_len + grab_extra)
        # Extract from ORIGINAL lq audio
        raw            = lq[lq_grab_start:lq_grab_end]

        if abs(sub_off) > 0.05 and raw.shape[0] > c_len:
            xs_src = np.arange(raw.shape[0], dtype=np.float64)
            xs_dst = np.clip(np.arange(c_len, dtype=np.float64) + sub_off,
                             0, raw.shape[0] - 1)
            chunk = np.stack([np.interp(xs_dst, xs_src, raw[:, ch])
                               for ch in range(n_channels)], axis=1)
        else:
            chunk = raw[:c_len]
            if chunk.shape[0] < c_len:
                chunk = np.pad(chunk, ((0, c_len - chunk.shape[0]), (0, 0)))

        if chunk.shape[0] != c_len:
            chunk = chunk[:c_len]
        return chunk

    for i, chunk_start in enumerate(chunk_starts):
        chunk_end = min(chunk_start + chunk_len, n_samples)
        c_len     = chunk_end - chunk_start
        if c_len < 1:
            break

        is_low_conf = bool(low_mask[i]) if i < len(low_mask) else False

        if is_low_conf:
            prev_conf = high_conf_idx[high_conf_idx < i]
            next_conf = high_conf_idx[high_conf_idx > i]

            if len(prev_conf) > 0 and len(next_conf) > 0:
                pi, ni = int(prev_conf[-1]), int(next_conf[0])
                p_start, n_start = chunk_starts[pi], chunk_starts[ni]

                p_chunk_src = extract_chunk(pi, p_start, min(chunk_len, n_samples - p_start))
                n_chunk_src = extract_chunk(ni, n_start, min(chunk_len, n_samples - n_start))

                p_off, n_off = chunk_start - p_start, chunk_start - n_start

                def slice_neighbour(src, off, length):
                    if off >= 0 and off < src.shape[0]:
                        sl = src[off:off + length]
                    elif off < 0:
                        avail = src[:length + off] if length + off > 0 else src[:0]
                        sl = np.pad(avail, ((max(0, -off), 0), (0, 0)))
                    else:
                        sl = np.zeros((0, n_channels))
                    if sl.shape[0] < length:
                        sl = np.pad(sl, ((0, length - sl.shape[0]), (0, 0)))
                    return sl[:length]

                p_audio = slice_neighbour(p_chunk_src, p_off, c_len)
                n_audio = slice_neighbour(n_chunk_src, n_off, c_len)

                t = np.linspace(0, 1, c_len)[:, np.newaxis]
                lq_chunk = p_audio * (1 - t) + n_audio * t
            elif len(prev_conf) > 0:
                pi = int(prev_conf[-1])
                p_start = chunk_starts[pi]
                src = extract_chunk(pi, p_start, min(chunk_len, n_samples - p_start))
                off = chunk_start - p_start
                lq_chunk = src[off:off + c_len] if off < src.shape[0] else np.zeros((c_len, n_channels))
                if lq_chunk.shape[0] < c_len:
                    lq_chunk = np.pad(lq_chunk, ((0, c_len - lq_chunk.shape[0]), (0, 0)))
            else:
                lq_chunk = extract_chunk(i, chunk_start, c_len)
        else:
            lq_chunk = extract_chunk(i, chunk_start, c_len)

        fi = min(cf_len, c_len // 4)
        fo = min(cf_len, c_len // 4)
        env = np.ones(c_len)
        if i > 0: env[:fi] = hann_fade(fi)
        if i < n_chunks - 1: env[-fo:] = hann_fade(fo)[::-1]

        out[chunk_start:chunk_end]    += lq_chunk * env[:, np.newaxis]
        weight[chunk_start:chunk_end] += env

        if n_chunks >= 10 and (i + 1) % max(1, n_chunks // 10) == 0:
            drift_smp = int(lq_starts[i]) - chunk_start
            print(f"    {100*(i+1)//n_chunks}%  chunk {i+1}/{n_chunks}  "
                  f"drift={drift_smp:+d}smp  conf={confidences[i]:.3f}"
                  + ("  [xfade]" if is_low_conf else ""))

    weight = np.maximum(weight, 1e-12)
    out /= weight[:, np.newaxis]
    return out

def normalize_to_dbfs(data: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """Peak-normalize audio to target_dbfs (default -1.0 dBFS)."""
    peak = np.max(np.abs(data))
    if peak < 1e-9:
        return data
    target_linear = 10 ** (target_dbfs / 20.0)
    return data * (target_linear / peak)

def main():
    parser = argparse.ArgumentParser(
        description="Align a vinyl LQ track to its CD HQ counterpart with Internal Proxy Leveling."
    )
    parser.add_argument("--hq", required=True, help="Path to HQ (CD) reference audio")
    parser.add_argument("--lq", required=True, help="Path to LQ (vinyl) audio to align")
    parser.add_argument("--out", required=True, help="Output path for aligned LQ audio")
    parser.add_argument("--anchor-ms", type=float, default=500.0, help="End anchor duration (ms)")
    parser.add_argument("--search-ppm", type=int, default=3000, help="±PPM search range")
    parser.add_argument("--search-steps", type=int, default=600, help="Kept for API compat")
    parser.add_argument("--chunk-ms", type=float, default=150.0, help="Micro-align chunk size (ms)")
    parser.add_argument("--crossfade-ms", type=float, default=8.0, help="Crossfade length (ms)")
    parser.add_argument("--max-drift-ms", type=float, default=50.0, help="Max drift search (ms)")
    parser.add_argument("--conf-threshold", type=float, default=0.05, help="Min xcorr confidence")
    parser.add_argument("--skip-global", action="store_true", help="Skip global resample")
    parser.add_argument("--skip-chunks", action="store_true", help="Skip micro-alignment")
    parser.add_argument("--normalize-db", type=float, default=-1.0, help="Peak-normalize output")
    parser.add_argument("--out-hq", type=str, default=None, help="Save normalized HQ")

    args = parser.parse_args()

    print("\n[1/4] Loading and Pre-processing...")
    hq, hq_sr = load_audio(args.hq)
    lq, lq_sr = load_audio(args.lq)

    if hq_sr != lq_sr:
        print(f"  Resampling LQ {lq_sr} -> {hq_sr} using high-quality sinc...")
        lq = sinc_resample(lq, lq.shape[0], int(lq.shape[0] * hq_sr / lq_sr))
        sr = hq_sr
    else:
        sr = hq_sr

    # Step A: Check and Fix Stereo Swaps
    lq = check_stereo_swap(hq, lq)

    # Step B: RMS Leveling (Body matching)
    lq = match_rms(hq, lq)

    if not args.skip_global:
        print("\n[2/4] Global sinc resample (end-to-end alignment using proxies) …")
        ratio = find_global_ratio(
            hq, lq, anchor_ms=args.anchor_ms, sr=sr,
            search_ppm=args.search_ppm, search_steps=args.search_steps
        )
        target_len = round(lq.shape[0] * ratio)
        lq = sinc_resample(lq, lq.shape[0], target_len)
    
    hq_len = hq.shape[0]
    lq_len = lq.shape[0]
    if lq_len > hq_len:
        lq = lq[:hq_len]
    elif lq_len < hq_len:
        lq = np.pad(lq, ((0, hq_len - lq_len), (0, 0)))

    if not args.skip_chunks:
        print("\n[3/4] Chunked micro-alignment (using proxies) …")
        lq_aligned = micro_align_chunks(
            hq, lq, sr, chunk_ms=args.chunk_ms,
            crossfade_ms=args.crossfade_ms, max_drift_ms=args.max_drift_ms,
            conf_threshold=args.conf_threshold
        )
    else:
        lq_aligned = lq

    print("\n[4/4] Saving output with Peak Normalization …")
    lq_aligned = normalize_to_dbfs(lq_aligned, args.normalize_db)
    save_audio(args.out, lq_aligned, sr)

    if args.out_hq:
        hq = normalize_to_dbfs(hq, args.normalize_db)
        save_audio(args.out_hq, hq, sr)

    print("\nDone.\n")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# TUI screen
# ---------------------------------------------------------------------------

def screen_align_audio(state: dict, console, _pick, _run_with_live_output, ROOT) -> None:
    from pathlib import Path
    import subprocess, sys

    INPUT_DIR  = ROOT / "input"
    OUTPUT_DIR = ROOT / "output"

    def _pick_wav(prompt: str, last_key: str) -> str | None:
        """Pick a WAV file from input/, recent state, or custom path."""
        candidates = sorted(INPUT_DIR.glob("*.wav")) + sorted(INPUT_DIR.glob("*.flac"))
        items = [p.name for p in candidates]
        last = state.get(last_key, "")
        custom_label = f"Custom path  [{last}]" if last else "Custom path..."
        items += [custom_label, "Back"]
        idx = _pick(prompt, items, hint="Enter=select  Esc=back")
        if idx is None or idx == len(items) - 1:
            return None
        if idx == len(items) - 2:
            path = console.input("Path to file: ").strip().strip('"')
            if not path:
                return None
            state[last_key] = path
            return path
        chosen = str(candidates[idx])
        state[last_key] = chosen
        return chosen

    hq_path = _pick_wav("Align Audio -- select HQ reference", "align_last_hq")
    if not hq_path:
        return

    lq_path = _pick_wav("Align Audio -- select LQ to align", "align_last_lq")
    if not lq_path:
        return

    from pathlib import Path as _Path
    default_out = str(OUTPUT_DIR / (_Path(lq_path).stem + "_aligned.wav"))
    out_str = console.input(f"Output path [{default_out}]: ").strip().strip('"')
    out_path = out_str if out_str else default_out

    # Optional: save normalized HQ alongside aligned LQ?
    save_hq_str = console.input("Also save normalized HQ? (y/N): ").strip().lower()
    out_hq = None
    if save_hq_str == "y":
        default_hq_out = str(OUTPUT_DIR / (_Path(hq_path).stem + "_normalized.wav"))
        hq_out_str = console.input(f"  HQ output path [{default_hq_out}]: ").strip().strip('"')
        out_hq = hq_out_str if hq_out_str else default_hq_out

    # Advanced params — offer defaults or let user tweak
    adv = _pick(
        "Align Audio -- options",
        ["Run with defaults", "Adjust parameters", "Back"],
        hint="Enter=select  Esc=back",
    )
    if adv is None or adv == 2:
        return

    extra_flags: list[str] = []
    if adv == 1:
        def _ask_float(prompt: str, default: float) -> float:
            val = console.input(f"  {prompt} [{default}]: ").strip()
            try:
                return float(val) if val else default
            except ValueError:
                return default

        def _ask_int(prompt: str, default: int) -> int:
            val = console.input(f"  {prompt} [{default}]: ").strip()
            try:
                return int(val) if val else default
            except ValueError:
                return default

        console.print("[dim]Press Enter to keep default value.[/]\n")
        anchor    = _ask_float("--anchor-ms   (anchor window size)", 500.0)
        ppm       = _ask_int(  "--search-ppm  (±PPM drift range)",   3000)
        chunk     = _ask_float("--chunk-ms    (micro-align chunk)",  150.0)
        xfade     = _ask_float("--crossfade-ms",                       8.0)
        drift     = _ask_float("--max-drift-ms",                      50.0)
        conf      = _ask_float("--conf-threshold",                    0.05)
        norm_db   = _ask_float("--normalize-db",                      -1.0)

        skip_global = console.input("  Skip global resample? (y/N): ").strip().lower() == "y"
        skip_chunks = console.input("  Skip micro-alignment? (y/N): ").strip().lower() == "y"

        extra_flags = [
            "--anchor-ms",   str(anchor),
            "--search-ppm",  str(ppm),
            "--chunk-ms",    str(chunk),
            "--crossfade-ms",str(xfade),
            "--max-drift-ms",str(drift),
            "--conf-threshold", str(conf),
            "--normalize-db",str(norm_db),
        ]
        if skip_global:
            extra_flags.append("--skip-global")
        if skip_chunks:
            extra_flags.append("--skip-chunks")

    script = str(ROOT / "utils" / "align_audio.py")
    cmd = [sys.executable, script,
           "--hq", hq_path,
           "--lq", lq_path,
           "--out", out_path,
           ] + extra_flags
    if out_hq:
        cmd += ["--out-hq", out_hq]

    _run_with_live_output(cmd, "Align Audio")
