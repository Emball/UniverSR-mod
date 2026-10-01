import logging
import random
from fractions import Fraction
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from audio_io import resample

log = logging.getLogger(__name__)


@dataclass
class GainAugCfg:
    enabled: bool = True
    prob: float = 0.5
    db_max: float = 1.5


@dataclass
class SimpleAugCfg:
    enabled: bool = True
    prob: float = 0.5


@dataclass
class Mp3AugCfg:
    enabled: bool = False
    prob: float = 0.5
    kbps_min: int = 64
    kbps_max: int = 256
    target: str = "lq"


@dataclass
class DeepGainAugCfg:
    enabled: bool = True
    prob: float = 0.03
    db_min: float = -10.0
    db_max: float = -6.0


@dataclass
class SilenceDipAugCfg:
    enabled: bool = True
    prob: float = 0.05
    max_hold_sec: float = 2.0
    short_ramp_ms: float = 10.0
    short_ramp_max_ms: float = 50.0
    medium_ramp_ms: float = 50.0
    medium_ramp_max_ms: float = 200.0
    long_ramp_ms: float = 200.0
    long_ramp_max_ms: float = 1000.0


@dataclass
class MidSideAugCfg:
    enabled: bool = False
    prob_mid: float = 0.1
    prob_side: float = 0.1


@dataclass
class PitchShiftAugCfg:
    enabled: bool = False
    prob: float = 0.5
    semitones_max: float = 1.5


@dataclass
class NoiseAugCfg:
    enabled: bool = False
    prob: float = 0.5
    sigma: float = 0.002


@dataclass
class AugmentationCfg:
    enabled: bool = True
    gain: GainAugCfg = field(default_factory=GainAugCfg)
    deep_gain: DeepGainAugCfg = field(default_factory=DeepGainAugCfg)
    polarity: SimpleAugCfg = field(default_factory=SimpleAugCfg)
    silence_dip: SilenceDipAugCfg = field(default_factory=SilenceDipAugCfg)
    mp3_degradation: Mp3AugCfg = field(default_factory=Mp3AugCfg)
    stereo_alternation: SimpleAugCfg = field(default_factory=SimpleAugCfg)
    mid_side_isolation: MidSideAugCfg = field(default_factory=MidSideAugCfg)
    pitch_shift: PitchShiftAugCfg = field(default_factory=PitchShiftAugCfg)
    noise: NoiseAugCfg = field(default_factory=NoiseAugCfg)


def _get(d, key, default):
    try:
        v = d[key]
    except (KeyError, TypeError, IndexError):
        return default
    return default if v is None else v


def parse_live_aug_cfg(raw) -> AugmentationCfg:
    if raw is None:
        return AugmentationCfg()
    live = _get(raw, "live", None)
    if live is not None:
        raw = live

    gain = _get(raw, "gain", {})
    dg = _get(raw, "deep_gain", {})
    pol = _get(raw, "polarity", {})
    sil = _get(raw, "silence_dip", {})
    mp3 = _get(raw, "mp3_degradation", {})
    alt = _get(raw, "stereo_alternation", {})
    ms = _get(raw, "mid_side_isolation", {})
    ps = _get(raw, "pitch_shift", {})
    ns = _get(raw, "noise", {})

    return AugmentationCfg(
        enabled=_get(raw, "enabled", True),
        gain=GainAugCfg(_get(gain, "enabled", True), _get(gain, "prob", 0.5), _get(gain, "db_max", 1.5)),
        deep_gain=DeepGainAugCfg(
            _get(dg, "enabled", True), _get(dg, "prob", 0.03),
            _get(dg, "db_min", -10.0), _get(dg, "db_max", -6.0),
        ),
        polarity=SimpleAugCfg(_get(pol, "enabled", True), _get(pol, "prob", 0.5)),
        silence_dip=SilenceDipAugCfg(
            enabled=_get(sil, "enabled", True),
            prob=_get(sil, "prob", 0.05),
            max_hold_sec=_get(sil, "max_hold_sec", 2.0),
            short_ramp_ms=_get(sil, "short_ramp_ms", 10.0),
            short_ramp_max_ms=_get(sil, "short_ramp_max_ms", 50.0),
            medium_ramp_ms=_get(sil, "medium_ramp_ms", 50.0),
            medium_ramp_max_ms=_get(sil, "medium_ramp_max_ms", 200.0),
            long_ramp_ms=_get(sil, "long_ramp_ms", 200.0),
            long_ramp_max_ms=_get(sil, "long_ramp_max_ms", 1000.0),
        ),
        mp3_degradation=Mp3AugCfg(
            enabled=_get(mp3, "enabled", False),
            prob=_get(mp3, "prob", 0.5),
            kbps_min=int(_get(mp3, "kbps_min", 64)),
            kbps_max=int(_get(mp3, "kbps_max", 256)),
            target=_get(mp3, "target", "lq"),
        ),
        stereo_alternation=SimpleAugCfg(_get(alt, "enabled", True), _get(alt, "prob", 1.0)),
        mid_side_isolation=MidSideAugCfg(
            _get(ms, "enabled", False), _get(ms, "prob_mid", 0.1), _get(ms, "prob_side", 0.1)
        ),
        pitch_shift=PitchShiftAugCfg(
            _get(ps, "enabled", False), _get(ps, "prob", 0.5), _get(ps, "semitones_max", 1.5)
        ),
        noise=NoiseAugCfg(_get(ns, "enabled", False), _get(ns, "prob", 0.5), _get(ns, "sigma", 0.002)),
    )


def parse_cached_aug_cfg(cached) -> Optional[AugmentationCfg]:
    if cached is None or not _get(cached, "enabled", False):
        return None

    def blk(name):
        return _get(cached, name, {})

    def frac(b, d=0.0):
        return float(_get(b, "fraction", d))

    g, pol, ps, ns, mp3, alt = (blk(n) for n in (
        "gain", "polarity", "pitch_shift", "noise", "mp3_degradation", "stereo_alternation"))
    return AugmentationCfg(
        enabled=True,
        gain=GainAugCfg(bool(_get(g, "enabled", False)), frac(g), float(_get(g, "db_max", 1.5))),
        deep_gain=DeepGainAugCfg(enabled=False),
        polarity=SimpleAugCfg(bool(_get(pol, "enabled", False)), frac(pol)),
        silence_dip=SilenceDipAugCfg(enabled=False),
        pitch_shift=PitchShiftAugCfg(
            bool(_get(ps, "enabled", False)), frac(ps), float(_get(ps, "semitones_max", 1.5))),
        noise=NoiseAugCfg(bool(_get(ns, "enabled", False)), frac(ns), float(_get(ns, "sigma", 0.002))),
        mp3_degradation=Mp3AugCfg(
            enabled=bool(_get(mp3, "enabled", False)),
            prob=frac(mp3, 0.5),
            kbps_min=int(_get(mp3, "kbps_min", 64)),
            kbps_max=int(_get(mp3, "kbps_max", 256)),
            target=_get(mp3, "target", "lq"),
        ),
        stereo_alternation=SimpleAugCfg(bool(_get(alt, "enabled", False)), frac(alt, 1.0)),
        mid_side_isolation=MidSideAugCfg(enabled=False),
    )


def build_cached_aug_fn(cached, sr: int) -> Optional[Callable]:
    cfg = parse_cached_aug_cfg(cached)
    if cfg is None:
        return None

    def apply(lq, hq):
        return augment_pair(lq, hq, cfg, sr=sr)

    return apply


_ffmpeg_ok: Optional[bool] = None


def _check_ffmpeg() -> bool:
    global _ffmpeg_ok
    if _ffmpeg_ok is None:
        try:
            import ffmpeg  # noqa: F401
            _ffmpeg_ok = True
        except ImportError:
            log.warning("ffmpeg-python not installed; mp3_degradation disabled")
            _ffmpeg_ok = False
    return _ffmpeg_ok


def pitch_shift_tensor(wav: torch.Tensor, semitones: float) -> torch.Tensor:
    # resample-based pitch+speed change; small rational ratio keeps the kernel tiny.
    # Output length scales by 1/factor; callers crop, so no padding here.
    frac = Fraction(2 ** (semitones / 12)).limit_denominator(64)
    return resample(wav, frac.numerator, frac.denominator).float()


def mp3_degrade_tensor(wav: torch.Tensor, kbps: int, sr: int) -> torch.Tensor:
    import ffmpeg

    n = wav.shape[-1]
    ch = wav.shape[0]

    def encode_decode(t):
        pcm = t.numpy().T.astype(np.float32).tobytes()
        mp3, _ = (
            ffmpeg.input("pipe:", format="f32le", ar=sr, ac=ch)
            .output("pipe:", format="mp3", audio_bitrate=f"{kbps}k", codec="libmp3lame")
            .run(input=pcm, capture_stdout=True, capture_stderr=True, quiet=True)
        )
        out, _ = (
            ffmpeg.input("pipe:", format="mp3")
            .output("pipe:", format="f32le", ar=sr, ac=ch)
            .run(input=mp3, capture_stdout=True, capture_stderr=True, quiet=True)
        )
        return torch.from_numpy(np.frombuffer(out, dtype=np.float32).reshape(-1, ch).T.copy())

    probe_len = 2048
    impulse = torch.zeros(ch, probe_len)
    impulse[:, 0] = 1.0
    probed = encode_decode(torch.cat([impulse, wav.float()], dim=-1))
    delay = int(probed[0, :probe_len * 2].abs().argmax().item())

    dec = encode_decode(wav.float())[:, delay:]
    if dec.shape[-1] >= n:
        dec = dec[:, :n]
    else:
        dec = torch.nn.functional.pad(dec, (0, n - dec.shape[-1]))
    return dec.float()


def silence_dip_envelope(n: int, sr: int, cfg: SilenceDipAugCfg) -> torch.Tensor:
    r = random.random()
    if r < 0.70:
        ramp_ms = random.uniform(cfg.short_ramp_ms, cfg.short_ramp_max_ms)
    elif r < 0.90:
        ramp_ms = random.uniform(cfg.medium_ramp_ms, cfg.medium_ramp_max_ms)
    else:
        ramp_ms = random.uniform(cfg.long_ramp_ms, cfg.long_ramp_max_ms)

    ramp = max(1, int(ramp_ms / 1000.0 * sr))
    hold = random.randint(1, max(1, int(cfg.max_hold_sec * sr)))
    dip = 2 * ramp + hold
    max_start = n - dip - 2
    if max_start <= 1:
        return torch.ones(1, n)

    s = random.randint(1, max_start)
    env = torch.ones(n)
    env[s:s + ramp] = torch.linspace(1.0, 0.0, ramp)
    env[s + ramp:s + ramp + hold] = 0.0
    env[s + ramp + hold:s + dip] = torch.linspace(0.0, 1.0, ramp)
    return env.unsqueeze(0)


def _ensure_mono(lq: torch.Tensor, hq: torch.Tensor, ch: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if lq.shape[0] > 1:
        c = min(ch, lq.shape[0] - 1)
        lq, hq = lq[c:c + 1], hq[c:c + 1]
    return lq, hq


def augment_pair(
    lq: torch.Tensor,
    hq: torch.Tensor,
    cfg: AugmentationCfg,
    sr: int = 48000,
    in_second_half: bool = False,
    forced_kbps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    half_ch = 1 if in_second_half else 0
    if not cfg.enabled:
        return _ensure_mono(lq, hq, random.randint(0, 1))

    ms_fired = False
    if cfg.mid_side_isolation.enabled and lq.shape[0] == 2:
        r = random.random()
        if r < cfg.mid_side_isolation.prob_mid:
            lq = (lq[0:1] + lq[1:2]) * 0.5
            hq = (hq[0:1] + hq[1:2]) * 0.5
            ms_fired = True
        elif r < cfg.mid_side_isolation.prob_mid + cfg.mid_side_isolation.prob_side:
            lq = (lq[0:1] - lq[1:2]) * 0.5
            hq = (hq[0:1] - hq[1:2]) * 0.5
            ms_fired = True

    if not ms_fired:
        ch = half_ch if cfg.stereo_alternation.enabled else random.randint(0, 1)
        lq, hq = _ensure_mono(lq, hq, ch)

    if cfg.gain.enabled and random.random() < cfg.gain.prob:
        scale = 10 ** (random.uniform(-cfg.gain.db_max, cfg.gain.db_max) / 20.0)
        lq, hq = lq * scale, hq * scale
        peak = max(lq.abs().max(), hq.abs().max())
        if peak > 1.0:
            lq, hq = lq / peak, hq / peak

    if cfg.polarity.enabled and in_second_half:
        lq, hq = -lq, -hq

    if cfg.deep_gain.enabled and random.random() < cfg.deep_gain.prob:
        scale = 10 ** (random.uniform(cfg.deep_gain.db_min, cfg.deep_gain.db_max) / 20.0)
        lq, hq = lq * scale, hq * scale

    if cfg.pitch_shift.enabled and random.random() < cfg.pitch_shift.prob:
        semis = random.uniform(-cfg.pitch_shift.semitones_max, cfg.pitch_shift.semitones_max)
        lq = pitch_shift_tensor(lq, semis)
        hq = pitch_shift_tensor(hq, semis)

    if cfg.noise.enabled and random.random() < cfg.noise.prob:
        noise = torch.randn_like(hq) * cfg.noise.sigma
        lq, hq = lq + noise, hq + noise

    if cfg.silence_dip.enabled and random.random() < cfg.silence_dip.prob:
        env = silence_dip_envelope(lq.shape[-1], sr, cfg.silence_dip)
        lq, hq = lq * env, hq * env

    if cfg.mp3_degradation.enabled and random.random() < cfg.mp3_degradation.prob and _check_ffmpeg():
        kbps = forced_kbps if forced_kbps is not None else random.randint(
            cfg.mp3_degradation.kbps_min, cfg.mp3_degradation.kbps_max)
        lq = mp3_degrade_tensor(lq, kbps, sr)
        if cfg.mp3_degradation.target == "both":
            hq = mp3_degrade_tensor(hq, kbps, sr)

    return lq, hq
