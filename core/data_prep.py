import concurrent.futures as cf
import hashlib
import json
import logging
import os
import random
import shutil
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from audio_io import SUPPORTED_EXTS, decode_to_wav_cache, load_audio, peak_normalize_pair, save_wav_f32
from augment import build_cached_aug_fn

log = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "usr", "cache")
DECODE_DIR = os.path.join(CACHE_DIR, "decoded")
CHUNK_CACHE_DIR = os.path.join(CACHE_DIR, "chunks")


def say(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class PrepCfg:
    sr: int = 48000
    chunk_sec: float = 3.0
    overlap: float = 0.5
    fixed_delay: Optional[int] = None
    cache_peak_dbfs: float = -1.0
    cached_aug_fn: Optional[Callable] = None

    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_sec * self.sr)

    @property
    def hop_samples(self) -> int:
        return max(1, int(self.chunk_samples * (1 - self.overlap)))

    @property
    def lq_trim(self) -> int:
        return self.fixed_delay if self.fixed_delay and self.fixed_delay > 0 else 0

    @property
    def hq_trim(self) -> int:
        return -self.fixed_delay if self.fixed_delay and self.fixed_delay < 0 else 0

    def manifest(self) -> dict:
        return {
            "sr": self.sr,
            "segment_sec": self.chunk_sec,
            "overlap": self.overlap,
            "fixed_delay": str(self.fixed_delay),
            "cache_peak_dbfs": self.cache_peak_dbfs,
        }


def _stems(d: str) -> dict:
    return {os.path.splitext(f)[0]: f for f in os.listdir(d)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS}


def _has_wavs(d: str) -> bool:
    return os.path.isdir(d) and any(f.endswith(".wav") for f in os.listdir(d))


def normalize_data_dir(src_root: str, split_name: str) -> bool:
    """Normalise src_root into LQ/ + HQ/. Accepts <song>_LQ/<song>_HQ subdirs,
    flat <song>_LQ.ext/<song>_HQ.ext files, or an existing LQ/ + HQ/ layout."""
    if not os.path.isdir(src_root):
        return False

    lq_dir = os.path.join(src_root, "LQ")
    hq_dir = os.path.join(src_root, "HQ")
    if os.path.isdir(lq_dir) and os.path.isdir(hq_dir) and _stems(lq_dir) and _stems(hq_dir):
        say(f"[data/{split_name}] LQ/ + HQ/ already present -- skipping normalization.")
        return True

    entries = os.listdir(src_root)
    subdirs = {e for e in entries if os.path.isdir(os.path.join(src_root, e))}
    lq_dirs = {d[:-3]: d for d in subdirs if d.upper().endswith("_LQ")}
    hq_dirs = {d[:-3]: d for d in subdirs if d.upper().endswith("_HQ")}
    dir_pairs = sorted(set(lq_dirs) & set(hq_dirs))

    lq_flat, hq_flat = {}, {}
    for e in entries:
        p = os.path.join(src_root, e)
        stem, ext = os.path.splitext(e)
        if not os.path.isfile(p) or ext.lower() not in SUPPORTED_EXTS:
            continue
        if stem.upper().endswith("_LQ"):
            lq_flat[stem[:-3]] = e
        elif stem.upper().endswith("_HQ"):
            hq_flat[stem[:-3]] = e
    file_pairs = sorted(set(lq_flat) & set(hq_flat))

    if not dir_pairs and not file_pairs:
        say(f"[data/{split_name}] WARNING: no _LQ/_HQ pairs found in {src_root}")
        return False

    os.makedirs(lq_dir, exist_ok=True)
    os.makedirs(hq_dir, exist_ok=True)

    for stem in dir_pairs:
        for tag, sub, dst in (("LQ", lq_dirs[stem], lq_dir), ("HQ", hq_dirs[stem], hq_dir)):
            sd = os.path.join(src_root, sub)
            for fname in sorted(os.listdir(sd)):
                if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS:
                    shutil.move(os.path.join(sd, fname), os.path.join(dst, f"{stem}_{fname}"))
            try:
                os.rmdir(sd)
            except OSError:
                pass
        say(f"[data/{split_name}]   normalized dir pair: {stem}")

    for stem in file_pairs:
        for fname, dst in ((lq_flat[stem], lq_dir), (hq_flat[stem], hq_dir)):
            shutil.move(os.path.join(src_root, fname),
                        os.path.join(dst, f"{stem}{os.path.splitext(fname)[1]}"))
        say(f"[data/{split_name}]   normalized file pair: {stem}")

    say(f"[data/{split_name}] Normalized {len(dir_pairs) + len(file_pairs)} pairs into LQ/ + HQ/")
    return True


def source_md5s(src_root: str) -> str:
    h = hashlib.md5()
    for sub in ("LQ", "HQ"):
        d = os.path.join(src_root, sub)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            path = os.path.join(d, fname)
            if not os.path.isfile(path):
                continue
            fh = hashlib.md5()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    fh.update(block)
            h.update(fname.encode())
            h.update(fh.hexdigest().encode())
    return h.hexdigest()


def chunk_cache_key(src_root: str, p: PrepCfg, aug_cfg=None, extra: str = "") -> str:
    params = dict(p.manifest())
    params["aug"] = None if aug_cfg is None else json.dumps(
        OmegaConf.to_container(aug_cfg, resolve=True) if OmegaConf.is_config(aug_cfg) else aug_cfg,
        sort_keys=True, default=str)
    params["extra"] = extra
    h = hashlib.md5()
    h.update(source_md5s(src_root).encode())
    h.update(json.dumps(params, sort_keys=True).encode())
    return h.hexdigest()[:16]


def chunk_cache_lookup(key: str, split: str) -> Optional[str]:
    d = os.path.join(CHUNK_CACHE_DIR, key, split)
    return d if _has_wavs(os.path.join(d, "LQ")) else None


def slice_and_save(lq: torch.Tensor, hq: torch.Tensor, stem: str, lq_out: str, hq_out: str,
                   p: PrepCfg, progress_cb: Optional[Callable] = None) -> List[str]:
    n = min(lq.shape[-1], hq.shape[-1])
    lq, hq = lq[:, :n], hq[:, :n]

    lq, hq = peak_normalize_pair(lq, hq, p.cache_peak_dbfs)

    cs, hop = p.chunk_samples, p.hop_samples
    total = max(0, (n - cs) // hop + 1)
    saved, start, idx = [], 0, 0
    while start + cs <= n:
        lq_c, hq_c = lq[:, start:start + cs], hq[:, start:start + cs]
        if p.cached_aug_fn is not None:
            lq_c, hq_c = p.cached_aug_fn(lq_c.clone(), hq_c.clone())
        fname = f"{stem}_{idx:04d}.wav"
        save_wav_f32(lq_c, os.path.join(lq_out, fname), p.sr)
        save_wav_f32(hq_c, os.path.join(hq_out, fname), p.sr)
        saved.append(fname)
        if progress_cb:
            progress_cb(idx + 1, total)
        start += hop
        idx += 1
    return saved


def chunk_split(src_root: str, dst_root: str, split_name: str, p: PrepCfg) -> int:
    lq_out = os.path.join(dst_root, "LQ")
    hq_out = os.path.join(dst_root, "HQ")
    manifest_path = os.path.join(dst_root, ".manifest.json")

    if _has_wavs(lq_out):
        ok = False
        try:
            with open(manifest_path) as f:
                ok = json.load(f) == p.manifest()
        except Exception:
            pass
        if ok:
            n = sum(1 for f in os.listdir(lq_out) if f.endswith(".wav"))
            say(f"[data/{split_name}] Already chunked ({n} pairs) -- skipping.")
            return n
        say(f"[data/{split_name}] Chunk parameters changed -- re-chunking.")

    if not normalize_data_dir(src_root, split_name):
        return 0

    lq_src, hq_src = os.path.join(src_root, "LQ"), os.path.join(src_root, "HQ")
    lq_files, hq_files = _stems(lq_src), _stems(hq_src)
    matched = sorted(set(lq_files) & set(hq_files))
    unmatched = (set(lq_files) | set(hq_files)) - set(matched)
    if unmatched:
        say(f"[data/{split_name}] WARNING: unmatched files (skipping): {sorted(unmatched)}")
    if not matched:
        say(f"[data/{split_name}] ERROR: no matched LQ/HQ pairs after normalization")
        return 0

    os.makedirs(lq_out, exist_ok=True)
    os.makedirs(hq_out, exist_ok=True)
    say(f"[data/{split_name}] Chunking {len(matched)} pairs at {p.sr} Hz -> {dst_root}")

    lock = threading.Lock()
    non_wav = [os.path.join(d, fs[s]) for d, fs in ((lq_src, lq_files), (hq_src, hq_files))
               for s in matched if os.path.splitext(fs[s])[1].lower() != ".wav"]
    if non_wav:
        say(f"[data/{split_name}] Decoding {len(non_wav)} non-WAV source(s) to native-rate cache...")
        with cf.ThreadPoolExecutor(max_workers=min(len(non_wav), 2)) as ex:
            list(ex.map(lambda f: decode_to_wav_cache(f, DECODE_DIR), non_wav))

    done = {"chunks": 0, "songs": 0}

    def work(stem: str) -> int:
        lq = load_audio(os.path.join(lq_src, lq_files[stem]), p.sr, DECODE_DIR, p.lq_trim)
        hq = load_audio(os.path.join(hq_src, hq_files[stem]), p.sr, DECODE_DIR, p.hq_trim)

        def progress(_d, _t):
            with lock:
                done["chunks"] += 1
                if done["chunks"] % 50 == 0:
                    say(f"[data/{split_name}] {done['songs']}/{len(matched)} songs  {done['chunks']} chunks")

        saved = slice_and_save(lq, hq, stem, lq_out, hq_out, p, progress)
        with lock:
            done["songs"] += 1
            say(f"[data/{split_name}]   {stem}: done ({len(saved)} chunks)  [{done['songs']}/{len(matched)}]")
        return len(saved)

    with cf.ThreadPoolExecutor(max_workers=min(len(matched), 2)) as pool:
        total = sum(pool.map(work, matched))

    say(f"[data/{split_name}] Done -- {total} chunk pairs -> {dst_root}")
    with open(manifest_path, "w") as f:
        json.dump(p.manifest(), f, indent=2)
    return total


def _pick_two_rms_clips(wav: torch.Tensor, clip_samples: int, sr: int, margin_sec: float = 5.0) -> List[int]:
    margin = int(margin_sec * sr)
    n = wav.shape[-1]
    lo, hi = margin, n - margin - clip_samples
    if hi <= lo:
        mid = max(0, n // 2 - clip_samples // 2)
        return [0, min(mid, max(0, n - clip_samples))]

    hop = max(1, clip_samples // 4)
    offsets = list(range(lo, hi + 1, hop)) or [lo]
    scores = sorted(((float(wav[s:s + clip_samples].pow(2).mean().sqrt()), s) for s in offsets),
                    key=lambda x: -x[0])
    best = scores[0][1]
    second = next((s for _, s in scores[1:] if abs(s - best) >= clip_samples), None)
    if second is None:
        cand = hi if best < (lo + hi) // 2 else lo
        second = max(lo, min(hi, cand))
    return sorted([best, second])


def extract_val_clips(src_root: str, dst_root: str, p: PrepCfg, clip_sec: float = 10.0) -> None:
    lq_src, hq_src = os.path.join(src_root, "LQ"), os.path.join(src_root, "HQ")
    if not os.path.isdir(lq_src) or not os.path.isdir(hq_src):
        return

    lq_files, hq_files = _stems(lq_src), _stems(hq_src)
    matched = sorted(set(lq_files) & set(hq_files))
    os.makedirs(os.path.join(dst_root, "LQ"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "HQ"), exist_ok=True)

    clip_samples = int(clip_sec * p.sr)
    say(f"[data/val] Extracting 2x{clip_sec:.0f}s highest-RMS clips from {len(matched)} val pair(s)")

    n = 0
    for i, stem in enumerate(matched):
        lq = load_audio(os.path.join(lq_src, lq_files[stem]), p.sr, DECODE_DIR, p.lq_trim)
        hq = load_audio(os.path.join(hq_src, hq_files[stem]), p.sr, DECODE_DIR, p.hq_trim)
        m = min(lq.shape[-1], hq.shape[-1])
        lq, hq = peak_normalize_pair(lq[:, :m], hq[:, :m], p.cache_peak_dbfs)

        starts = _pick_two_rms_clips(hq.mean(dim=0), clip_samples, p.sr)
        for k, s in enumerate(starts):
            lq_c, hq_c = lq[:, s:s + clip_samples], hq[:, s:s + clip_samples]
            if lq_c.shape[-1] < clip_samples:
                lq_c = torch.nn.functional.pad(lq_c, (0, clip_samples - lq_c.shape[-1]))
            if hq_c.shape[-1] < clip_samples:
                hq_c = torch.nn.functional.pad(hq_c, (0, clip_samples - hq_c.shape[-1]))
            name = f"{stem}_clip{k}.wav"
            save_wav_f32(lq_c, os.path.join(dst_root, "LQ", name), p.sr)
            save_wav_f32(hq_c, os.path.join(dst_root, "HQ", name), p.sr)
            say(f"[data/val]   {stem} clip{k}: {clip_sec:.0f}s @ {s // p.sr}s  [{i + 1}/{len(matched)}]")
            n += 1
    say(f"[data/val] Done -- {n} clip files -> {dst_root}")


def _bootstrap_val_from_train(train_chunks: str, val_dir: str, n_chunks: int) -> None:
    train_lq, train_hq = os.path.join(train_chunks, "LQ"), os.path.join(train_chunks, "HQ")
    if not _has_wavs(train_lq):
        say("[data/val] No train chunks available for val bootstrap -- skipping.")
        return

    by_song = defaultdict(list)
    for fname in sorted(os.listdir(train_lq)):
        if not fname.endswith(".wav"):
            continue
        parts = fname.rsplit("_", 1)
        key = parts[0] if len(parts) == 2 and parts[1][:-4].isdigit() else fname
        by_song[key].append(fname)

    songs = list(by_song)
    random.shuffle(songs)
    iters = {s: iter(random.sample(by_song[s], len(by_song[s]))) for s in songs}
    selected = []
    while len(selected) < n_chunks:
        progressed = False
        for s in songs:
            if len(selected) >= n_chunks:
                break
            try:
                selected.append(next(iters[s]))
                progressed = True
            except StopIteration:
                pass
        if not progressed:
            break

    os.makedirs(os.path.join(val_dir, "LQ"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "HQ"), exist_ok=True)
    for fname in selected:
        shutil.copy2(os.path.join(train_lq, fname), os.path.join(val_dir, "LQ", fname))
        shutil.copy2(os.path.join(train_hq, fname), os.path.join(val_dir, "HQ", fname))
    say(f"[data/val] Bootstrapped {len(selected)} val chunks from {len(songs)} training songs (copied, not moved).")


def prepare_data(cfg: DictConfig) -> None:
    datas = cfg.datas
    align = getattr(datas, "align_data", False)
    fixed_delay = int(align) if isinstance(align, int) and not isinstance(align, bool) and align != 0 else None

    aug_raw = getattr(datas, "augmentation", None)
    cached_raw = getattr(aug_raw, "cached", None) if aug_raw is not None else None

    sr = int(getattr(datas, "sr", 48000))
    p = PrepCfg(
        sr=sr,
        chunk_sec=float(getattr(datas, "segment_sec", 3.0)),
        overlap=float(getattr(datas, "overlap", 0.5)),
        fixed_delay=fixed_delay,
        cache_peak_dbfs=float(getattr(datas, "cache_peak_dbfs", -1.0)),
        cached_aug_fn=build_cached_aug_fn(cached_raw, sr),
    )

    data_root = os.path.join(REPO_ROOT, "data", cfg.exp.name)
    data_train, data_val = os.path.join(data_root, "train"), os.path.join(data_root, "val")
    normalize_data_dir(data_train, "train")
    normalize_data_dir(data_val, "val")

    cached_cfg_for_key = cached_raw if (cached_raw is not None and getattr(cached_raw, "enabled", False)) else None
    train_key = chunk_cache_key(data_train, p, cached_cfg_for_key)
    train_chunks = chunk_cache_lookup(train_key, "train")
    if train_chunks:
        say(f"[data/train] Cache hit ({train_key[:8]}...) -- skipping chunking.")
    else:
        train_chunks = os.path.join(CHUNK_CACHE_DIR, train_key, "train")
        chunk_split(data_train, train_chunks, "train", p)

    clip_sec = 10.0
    val_key = chunk_cache_key(data_val, p, None, extra=f"valclip_2x{clip_sec:.0f}s_rms")
    val_dir = os.path.join(CHUNK_CACHE_DIR, val_key, "val")
    if _has_wavs(os.path.join(val_dir, "LQ")):
        say(f"[data/val] Cache hit ({val_key[:8]}...) -- skipping clip extraction.")
    else:
        extract_val_clips(data_val, val_dir, p, clip_sec)

    with open_dict(cfg):
        cfg.datas.train_dir = train_chunks
        cfg.datas.eval_dir = val_dir

    if not _has_wavs(os.path.join(val_dir, "LQ")):
        _bootstrap_val_from_train(train_chunks, val_dir, int(datas.get("val_bootstrap_chunks", 50)))
