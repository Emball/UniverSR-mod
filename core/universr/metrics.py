import logging

import numpy as np
import torch

log = logging.getLogger("universr.metrics")

_visqol_api = None
_visqol_state = None


def _flat(x):
    return x.detach().float().cpu().reshape(-1, x.shape[-1])


def sisdr(est, ref, eps=1e-8):
    e = _flat(est).double()
    r = _flat(ref).double()
    n = min(e.shape[-1], r.shape[-1])
    e, r = e[..., :n], r[..., :n]
    e = e - e.mean(-1, keepdim=True)
    r = r - r.mean(-1, keepdim=True)
    proj = (e * r).sum(-1, keepdim=True) / (r.pow(2).sum(-1, keepdim=True) + eps) * r
    noise = e - proj
    val = 10 * torch.log10((proj.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))
    return float(val.mean())


def _mag(x, n_fft, hop):
    win = torch.hann_window(n_fft)
    return torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=win, return_complex=True).abs()


def hfnr(est, ref, sr=48000, lo_hz=8000.0, hi_hz=22000.0):
    n_fft, hop = 2048, 512
    bin_lo = int(lo_hz / (sr / n_fft))
    bin_hi = min(int(hi_hz / (sr / n_fft)), n_fft // 2)

    def flatness(x):
        band = _mag(_flat(x), n_fft, hop)[:, bin_lo:bin_hi, :].clamp(min=1e-10)
        return float((band.log().mean() - band.mean().log()).exp())

    eps = 1e-8
    return (flatness(est) + eps) / (flatness(ref) + eps)


def lsd(est, ref, cutoff_hz, sr=48000, n_fft=1024, hop=256):
    e = _flat(est)
    r = _flat(ref)
    n = min(e.shape[-1], r.shape[-1])
    pe = torch.log10(_mag(e[..., :n], n_fft, hop).square().clamp(min=1e-6))
    pr = torch.log10(_mag(r[..., :n], n_fft, hop).square().clamp(min=1e-6))
    cut = int(round(float(cutoff_hz) / (sr / n_fft)))

    def dist(a, b):
        if a.shape[-2] == 0:
            return float("nan")
        return float((a - b).square().mean(dim=-2).sqrt().mean())

    return dist(pe, pr), dist(pe[:, cut:], pr[:, cut:]), dist(pe[:, :cut], pr[:, :cut])


def _get_visqol():
    global _visqol_api, _visqol_state
    if _visqol_state is False:
        return None
    if _visqol_api is not None:
        return _visqol_api
    try:
        from visqol import VisqolApi
        api = VisqolApi()
        api.create(mode="audio")
        _visqol_api, _visqol_state = api, True
        log.info("ViSQOL loaded")
        return api
    except Exception as e:
        _visqol_state = False
        log.warning("ViSQOL unavailable (%s); ViSQOL scores will be skipped", e)
        return None


VISQOL_SR = 48000


def visqol(est, ref, sr=48000):
    api = _get_visqol()
    if api is None:
        return None
    try:
        e = est.detach().float().cpu().reshape(-1)
        r = ref.detach().float().cpu().reshape(-1)
        if sr != VISQOL_SR:
            from audio_io import resample
            e, r = resample(e[None], sr, VISQOL_SR)[0], resample(r[None], sr, VISQOL_SR)[0]
        e, r = e.numpy().astype(np.float64), r.numpy().astype(np.float64)
        pad = np.zeros(int(0.5 * VISQOL_SR), dtype=np.float64)
        res = api.measure_from_arrays(np.concatenate([pad, r, pad]), np.concatenate([pad, e, pad]), VISQOL_SR)
        return float(res.moslqo)
    except Exception as e:
        log.warning("ViSQOL measurement failed: %s", e)
        return None
