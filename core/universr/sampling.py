import logging

import torch
import torch.nn.functional as F

from universr.utils.spectral_ops import AmplitudeCompressedComplexSTFT

log = logging.getLogger("universr.sampling")


def make_transform(cfg):
    return AmplitudeCompressedComplexSTFT(**cfg)


def hz_to_cutoff_bins(cutoff_hz, sample_rate, n_fft):
    return torch.as_tensor(cutoff_hz, dtype=torch.float32) * n_fft / sample_rate


def to_spec(transform, wav, pad_to_hop=False):
    """[B,1,T] waveform -> [B,2,F,frames] compressed STFT, Nyquist bin dropped; always fp32.

    pad_to_hop zero-pads the tail to a multiple of the hop so the inverse never relies on the
    near-zero tail of the final window; training leaves it off to keep upstream framing.
    """
    hop = transform.complex_stft.hop_length
    if pad_to_hop and wav.shape[-1] % hop:
        wav = F.pad(wav, [0, (-wav.shape[-1]) % hop])
    with torch.autocast(device_type=wav.device.type, enabled=False):
        spec = transform(wav.float())
        real = torch.view_as_real(spec.squeeze(1)).permute(0, 3, 1, 2)
    return real[:, :, :-1, :].contiguous()


def to_wave(transform, spec, length=None):
    """[B,2,F,frames] -> [B,1,T]; inverts at the natural length, then crops or zero-pads to length."""
    hop = transform.complex_stft.hop_length
    natural = (spec.shape[-1] - 1) * hop
    with torch.autocast(device_type=spec.device.type, enabled=False):
        spec = F.pad(spec.float(), [0, 0, 0, 1])
        cplx = torch.view_as_complex(spec.permute(0, 2, 3, 1).contiguous())
        wav = transform.invert(cplx, orig_length=natural).unsqueeze(1)
    if length is not None and length != natural:
        wav = wav[..., :length] if length < natural else F.pad(wav, [0, length - natural])
    return wav


def assemble(Y, x1, cutoff_bins, gen_start_bin, keep_lq_below_cutoff):
    """Full-band spectrum from the LQ spectrum Y and generated region x1; per-sample cutoff."""
    full = torch.cat([Y[:, :, :gen_start_bin], x1.to(Y.dtype)], dim=2)
    if keep_lq_below_cutoff:
        idx = torch.arange(Y.shape[2], device=Y.device)
        mask = idx[None, :] < cutoff_bins.round().to(Y.device)[:, None]
        full = torch.where(mask[:, None, :, None], Y, full)
    return full


@torch.no_grad()
def ode_sample(model, y, cutoff_bins, num_frames, steps=4, guidance=1.5, seed=None):
    """Midpoint-method flow sampler with classifier-free guidance, in the generated region."""
    B = y.shape[0]
    dev = y.device
    gen = None
    if seed is not None:
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))
    x = torch.randn(B, 2, model.hr_freq_bins, num_frames, device=dev, generator=gen)
    ts = torch.linspace(0, 1, int(steps) + 1).tolist()
    use_cfg = guidance is not None and guidance != 0

    def field(xt, t):
        tt = torch.full((B,), t, device=dev)
        cond = model(xt, tt, y, cutoff_bins).float()
        if not use_cfg:
            return cond
        unc = model(xt, tt, None, cutoff_bins).float()
        return (1 - guidance) * unc + guidance * cond

    for t0, t1 in zip(ts[:-1], ts[1:]):
        h = t1 - t0
        mid = x + 0.5 * h * field(x, t0)
        x = x + h * field(mid, t0 + 0.5 * h)
    return x


@torch.no_grad()
def restore(model, transform, lq, cutoff_hz, sr, steps=4, guidance=1.5,
            keep_lq_below_cutoff=None, seed=None, amp_dtype=None):
    """lq [B,1,N] -> restored [B,1,N] fp32."""
    n_fft = transform.complex_stft.n_fft
    length = lq.shape[-1]
    cb = hz_to_cutoff_bins(cutoff_hz, sr, n_fft).to(lq.device).reshape(-1)
    if cb.numel() == 1:
        cb = cb.expand(lq.shape[0])
    if keep_lq_below_cutoff is None:
        keep_lq_below_cutoff = not model.aligned_input
    Y = to_spec(transform, lq, pad_to_hop=True)
    with torch.autocast(device_type=lq.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        x1 = ode_sample(model, Y, cb, Y.shape[-1], steps, guidance, seed)
    full = assemble(Y, x1, cb, model.gen_start_bin, keep_lq_below_cutoff)
    return to_wave(transform, full, length)


@torch.no_grad()
def restore_long(model, transform, lq, cutoff_hz, sr, chunk_sec=None, overlap_sec=0.5, **kw):
    """Chunked overlap-add restoration of [1,N] or [B,1,N]; chunk_sec=None runs in one pass."""
    squeeze = lq.ndim == 2
    if squeeze:
        lq = lq.unsqueeze(0)
    N = lq.shape[-1]
    if chunk_sec is None or N <= int(chunk_sec * sr):
        out = restore(model, transform, lq, cutoff_hz, sr, **kw)
        return out[0] if squeeze else out
    chunk = int(chunk_sec * sr)
    hop = max(1, chunk - int(overlap_sec * sr))
    out = torch.zeros(lq.shape[0], 1, N, device=lq.device)
    wsum = torch.zeros(N, device=lq.device)
    start = 0
    while start < N:
        end = min(start + chunk, N)
        piece = restore(model, transform, lq[..., start:end], cutoff_hz, sr, **kw)
        n = end - start
        w = torch.hann_window(n, periodic=False, device=lq.device) if n >= 2 else torch.ones(n, device=lq.device)
        out[..., start:end] += piece[..., :n] * w
        wsum[start:end] += w
        if end == N:
            break
        start += hop
    out = out / wsum.clamp(min=1e-8)
    return out[0] if squeeze else out
