"""
Paired LQ/HQ datamodule for UniverSR finetuning.

chunks/{train,val}/{LQ,HQ}/<stem>_NNNN.wav are produced by train.py's prepare_data()
at the configured sample rate. Every item is a single channel; stereo sources are
reduced by stereo_alternation (train) or by file-index parity (val).
"""

import logging
import os
import random
import re
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from augment import AugmentationCfg, augment_pair, parse_live_aug_cfg
from cutoff import resolve_cutoff_hz

log = logging.getLogger(__name__)

SR = 48000

_warned_resample = False

CutoffSpec = Union[None, str, float, Sequence[float]]


def load_wav(path: str, target_sr: int = SR) -> torch.Tensor:
    global _warned_resample
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        if not _warned_resample:
            log.warning("resampling %s from %d to %d at load time; chunk cache should already be at the target rate",
                        path, sr, target_sr)
            _warned_resample = True
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 2:
        wav = wav[:2]
    return wav


def peak_normalize_pair(lq: torch.Tensor, hq: torch.Tensor, target_dbfs: float):
    peak = max(lq.abs().max().item(), hq.abs().max().item())
    if peak < 1e-4:
        return lq, hq
    scale = (10 ** (target_dbfs / 20.0)) / peak
    return lq * scale, hq * scale


def random_crop_pair(lq: torch.Tensor, hq: torch.Tensor, n: int):
    length = min(lq.shape[-1], hq.shape[-1])
    lq, hq = lq[..., :length], hq[..., :length]
    if length < n:
        pad = n - length
        return torch.nn.functional.pad(lq, (0, pad)), torch.nn.functional.pad(hq, (0, pad))
    s = random.randint(0, length - n)
    return lq[..., s:s + n], hq[..., s:s + n]


def get_matched_pairs(lq_dir: str, hq_dir: str) -> List[Tuple[str, str]]:
    lq = {os.path.splitext(f)[0]: os.path.join(lq_dir, f) for f in os.listdir(lq_dir) if f.endswith(".wav")}
    hq = {os.path.splitext(f)[0]: os.path.join(hq_dir, f) for f in os.listdir(hq_dir) if f.endswith(".wav")}
    matched = sorted(set(lq) & set(hq))
    if set(lq) - set(hq):
        log.warning("LQ files with no HQ match (skipping): %s", sorted(set(lq) - set(hq)))
    if set(hq) - set(lq):
        log.warning("HQ files with no LQ match (skipping): %s", sorted(set(hq) - set(lq)))
    if not matched:
        raise RuntimeError(f"No matched pairs found in {lq_dir} and {hq_dir}")
    return [(lq[s], hq[s]) for s in matched]


def _cutoff_tensor(spec: CutoffSpec, lq: torch.Tensor, sr: int, rng=None) -> torch.Tensor:
    hz = resolve_cutoff_hz(spec, lq.numpy(), sr, rng)
    return torch.tensor(hz, dtype=torch.float32)


class ChunkedPairDataset(Dataset):
    def __init__(
        self,
        chunks_dir: str,
        sr: int = SR,
        aug_cfg: Optional[AugmentationCfg] = None,
        label: str = "Training",
        crop_samples: Optional[int] = 32767,
        cutoff_hz: CutoffSpec = "auto",
        train_peak_dbfs: Optional[Sequence[float]] = (-6.0, -1.0),
        val_peak_dbfs: Optional[float] = -3.0,
    ):
        self._is_val = label == "Validation"
        self.pairs = get_matched_pairs(os.path.join(chunks_dir, "LQ"), os.path.join(chunks_dir, "HQ"))
        self.sr = sr
        self.aug_cfg = aug_cfg or AugmentationCfg()
        self.crop_samples = int(crop_samples) if crop_samples else None
        self.cutoff_hz = cutoff_hz
        self.train_peak_dbfs = list(train_peak_dbfs) if train_peak_dbfs else None
        self.val_peak_dbfs = val_peak_dbfs

        by_stem = defaultdict(list)
        for i, (lq_path, _) in enumerate(self.pairs):
            stem = re.sub(r"_\d{4}$", "", os.path.splitext(os.path.basename(lq_path))[0])
            by_stem[stem].append(i)
        self._in_second_half = [False] * len(self.pairs)
        for indices in by_stem.values():
            indices = sorted(indices)
            mid = len(indices) // 2
            for k, gi in enumerate(indices):
                self._in_second_half[gi] = k >= mid

        if self._is_val:
            log.info("Validation dataset: %d clip pairs", len(self.pairs))
        else:
            a = self.aug_cfg
            log.info("Training dataset: %d chunk pairs, crop=%s, cutoff=%s, aug=%s "
                     "(alt=%s pol=%s gain=%s deep=%s dip=%s mp3=%s ms=%s pitch=%s noise=%s)",
                     len(self.pairs), self.crop_samples, cutoff_hz, a.enabled,
                     a.stereo_alternation.enabled, a.polarity.enabled, a.gain.enabled,
                     a.deep_gain.enabled, a.silence_dip.enabled, a.mp3_degradation.enabled,
                     a.mid_side_isolation.enabled, a.pitch_shift.enabled, a.noise.enabled)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        lq_path, hq_path = self.pairs[idx]
        lq = load_wav(lq_path, self.sr)
        hq = load_wav(hq_path, self.sr)
        n = min(lq.shape[-1], hq.shape[-1])
        lq, hq = lq[..., :n], hq[..., :n]

        if self._is_val:
            ch = min(idx % 2, lq.shape[0] - 1)
            lq, hq = lq[ch:ch + 1], hq[ch:ch + 1]
            if self.val_peak_dbfs is not None:
                lq, hq = peak_normalize_pair(lq, hq, self.val_peak_dbfs)
            cut = _cutoff_tensor(self.cutoff_hz, lq, self.sr, np.random.default_rng(idx))
            song_key = os.path.splitext(os.path.basename(lq_path))[0]
            return hq, lq, idx, song_key, cut

        lq, hq = augment_pair(lq, hq, self.aug_cfg, sr=self.sr, in_second_half=self._in_second_half[idx])
        cut = _cutoff_tensor(self.cutoff_hz, lq, self.sr)
        if self.crop_samples:
            lq, hq = random_crop_pair(lq, hq, self.crop_samples)
        if self.train_peak_dbfs:
            lq, hq = peak_normalize_pair(lq, hq, random.uniform(*self.train_peak_dbfs))
        return hq, lq, cut


class FullLengthPairDataset(Dataset):
    def __init__(
        self,
        eval_dir: str,
        sr: int = SR,
        segment_sec: float = 2.0,
        cutoff_hz: CutoffSpec = "auto",
        val_peak_dbfs: Optional[float] = -3.0,
    ):
        self.pairs = get_matched_pairs(os.path.join(eval_dir, "LQ"), os.path.join(eval_dir, "HQ"))
        self.sr = sr
        self.segment_samples = int(segment_sec * sr)
        self.cutoff_hz = cutoff_hz
        self.val_peak_dbfs = val_peak_dbfs

        self.index = []
        for pair_idx, (lq_path, hq_path) in enumerate(self.pairs):
            li, hi = torchaudio.info(lq_path), torchaudio.info(hq_path)
            if li.sample_rate != sr or hi.sample_rate != sr:
                log.warning("%s is %d Hz, expected %d; frame offsets are in file samples",
                            lq_path, li.sample_rate, sr)
            m = min(li.num_frames, hi.num_frames)
            s = 0
            while s + self.segment_samples <= m:
                self.index.append((pair_idx, s))
                s += self.segment_samples
        log.info("Eval dataset: %d files -> %d segments", len(self.pairs), len(self.index))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        pair_idx, start = self.index[idx]
        lq_path, hq_path = self.pairs[pair_idx]
        lq, _ = torchaudio.load(lq_path, frame_offset=start, num_frames=self.segment_samples)
        hq, _ = torchaudio.load(hq_path, frame_offset=start, num_frames=self.segment_samples)
        ch = min(pair_idx % 2, lq.shape[0] - 1)
        lq, hq = lq[ch:ch + 1], hq[min(ch, hq.shape[0] - 1):min(ch, hq.shape[0] - 1) + 1]
        if self.val_peak_dbfs is not None:
            lq, hq = peak_normalize_pair(lq, hq, self.val_peak_dbfs)
        cut = _cutoff_tensor(self.cutoff_hz, lq, self.sr, np.random.default_rng(idx))
        song_key = os.path.splitext(os.path.basename(lq_path))[0]
        return hq, lq, idx, song_key, cut


class PairedAudioDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        eval_dir: str,
        sr: int = SR,
        segment_sec: float = 2.0,
        crop_samples: Optional[int] = 32767,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        augmentation: Optional[dict] = None,
        cutoff_hz: CutoffSpec = "auto",
        train_peak_dbfs: Optional[Sequence[float]] = (-6.0, -1.0),
        val_peak_dbfs: Optional[float] = -3.0,
        val_bootstrap_chunks: int = 50,
        **kwargs,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.eval_dir = eval_dir
        self.sr = sr
        self.segment_sec = segment_sec
        self.crop_samples = crop_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.aug_cfg = parse_live_aug_cfg(augmentation)
        self.cutoff_hz = cutoff_hz
        self.train_peak_dbfs = train_peak_dbfs
        self.val_peak_dbfs = val_peak_dbfs
        self.val_bootstrap_chunks = val_bootstrap_chunks
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if self.data_train is None:
            self.data_train = ChunkedPairDataset(
                self.train_dir, self.sr, self.aug_cfg, "Training",
                self.crop_samples, self.cutoff_hz, self.train_peak_dbfs, self.val_peak_dbfs)
        if self.data_val is None:
            self.data_val = ChunkedPairDataset(
                self.eval_dir, self.sr, None, "Validation",
                None, self.cutoff_hz, None, self.val_peak_dbfs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
