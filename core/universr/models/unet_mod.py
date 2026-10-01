import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
from torch.utils.checkpoint import checkpoint

from universr.models.unet import (
    ConditionalVectorFieldModel,
    ConditioningEncoder2D,
    DecoderBlock,
    EncoderBlock,
    FrequencyPositionalEmbedding,
    GRN,
    LayerNorm,
    Midcoder,
    SinusoidalTimeEmbedding,
)

log = logging.getLogger("universr.model")

PRETRAINED_ANCHOR_BINS = (80, 128, 170, 256)


def hz_to_bins(hz, sample_rate, n_fft):
    return hz * n_fft / sample_rate


def _run(module, ckpt, *args):
    if ckpt and torch.is_grad_enabled():
        return checkpoint(module, *args, use_reentrant=False)
    return module(*args)


class FastGRN(GRN):
    """GRN with identical maths and the same parameters, but dtype-preserving: the original promotes the
    4x-wide activation to fp32 under AMP (fp32 Nx times fp16 x) and makes several full-size temporaries."""

    def forward(self, x):
        gx = torch.linalg.vector_norm(x, ord=2, dim=(1, 2), keepdim=True, dtype=torch.float32)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        scale = (self.gamma.float() * nx + 1.0).to(x.dtype)
        return x * scale + self.beta.to(x.dtype)


class FastLayerNorm(LayerNorm):
    """channels_first LayerNorm through the fused kernel instead of ~8 elementwise passes."""

    def forward(self, x):
        if self.data_format == "channels_first":
            y = F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps)
            return y.permute(0, 3, 1, 2)
        return super().forward(x)


class BandwidthEmbedding(nn.Module):
    """Piecewise-linear embedding over cutoff bins; clamped (held) outside the anchor range."""

    def __init__(self, anchor_bins, dim):
        super().__init__()
        a = torch.tensor(list(anchor_bins), dtype=torch.float32)
        if len(a) < 2 or not bool((a[1:] > a[:-1]).all()):
            raise ValueError(f"bw_anchor_bins must be strictly increasing with >=2 entries: {anchor_bins}")
        self.register_buffer("anchors", a, persistent=False)
        self.weight = nn.Parameter(torch.randn(len(a), dim))

    def forward(self, bins):
        b = bins.to(self.weight.dtype)
        k = self.anchors.numel()
        hi = torch.bucketize(b, self.anchors).clamp(1, k - 1)
        lo = hi - 1
        a_lo, a_hi = self.anchors[lo], self.anchors[hi]
        frac = ((b - a_lo) / (a_hi - a_lo)).clamp(0.0, 1.0).unsqueeze(-1)
        return self.weight[lo] * (1 - frac) + self.weight[hi] * frac


class MaskedConditioningEncoder(ConditioningEncoder2D):
    """Same parameters as the original encoder; padded bins are masked out of the frequency mean."""

    ckpt = False

    def forward(self, y_lr, f_emb_lr, bw_emb, mask):
        gamma, beta = torch.chunk(self.film_generator(f_emb_lr), 2, dim=-1)
        gamma = rearrange(gamma, 'f c -> 1 c f 1')
        beta = rearrange(beta, 'f c -> 1 c f 1')
        z = self.head(y_lr * gamma + beta)
        g, b = torch.chunk(self.sr_adapter(bw_emb), 2, dim=-1)
        z = z * g[:, :, None, None] + b[:, :, None, None]
        z = z * mask
        for blk in self.blocks:
            z = _run(blk, self.ckpt, z) * mask
        return z.sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)


class ConvNeXtUNetCondMod(ConditionalVectorFieldModel):
    def __init__(self, in_channels=2, out_channels=2,
                 dims=(64, 128, 256, 512), depths=(2, 2, 2, 4),
                 drop_path=0., time_dim=128, cond_dim=256,
                 total_freq_bins=512, gen_start_bin=80,
                 feature_enc_layers=10, cond_dropout_prob=0.1,
                 bw_anchor_bins=PRETRAINED_ANCHOR_BINS,
                 aligned_input=False, grad_checkpoint=False, fast_ops=True, channels_last=False):
        super().__init__()
        dims, depths = list(dims), list(depths)
        self.strides = 2 ** len(dims)
        self.in_channels = in_channels
        self.total_freq_bins = total_freq_bins
        self.gen_start_bin = gen_start_bin
        self.hr_freq_bins = total_freq_bins - gen_start_bin
        self.cond_dropout_prob = cond_dropout_prob
        self.cond_dim = cond_dim
        self.aligned_input = aligned_input
        self.grad_checkpoint = grad_checkpoint
        self.bw_anchor_bins = tuple(bw_anchor_bins)
        self.aligned_extra = (in_channels + 1) if aligned_input else 0

        self.time_embedder = SinusoidalTimeEmbedding(dim=time_dim)
        self.bw_embedder = BandwidthEmbedding(self.bw_anchor_bins, cond_dim)
        self.uncond_emb = nn.Parameter(torch.randn(cond_dim))
        self.sr_projector = nn.Linear(cond_dim, time_dim)
        self.freq_pos_enc = FrequencyPositionalEmbedding(num_bins=total_freq_bins, emb_dim=cond_dim)
        self.film_generator = nn.Linear(cond_dim, cond_dim * 2)
        self.conditioning_encoder = MaskedConditioningEncoder(cond_dim=cond_dim, num_blocks=feature_enc_layers)
        self.conditioning_encoder.ckpt = grad_checkpoint

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels + cond_dim + self.aligned_extra, dims[0], kernel_size=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(depths)):
            dim_out = dims[i + 1] if i + 1 < len(dims) else dims[i]
            self.encoders.append(EncoderBlock(dims[i], dim_out, depths[i], drop_path, time_dim))
        self.midcoder = Midcoder(dims[-1], depths[-1], drop_path, time_dim)
        for i in reversed(range(len(depths))):
            dim_in = dims[i + 1] if i + 1 < len(dims) else dims[i]
            self.decoders.append(DecoderBlock(dim_in, dims[i], depths[i], drop_path, time_dim))
        self.final_conv = nn.Conv2d(dims[0], out_channels, kernel_size=1)

        self.apply(self._init_weights)
        if self.aligned_extra:
            self.init_conv[0].weight.data[:, in_channels + cond_dim:] = 0
        self.channels_last = bool(channels_last)
        if fast_ops:
            for m in self.modules():
                if type(m) is GRN:
                    m.__class__ = FastGRN
                elif type(m) is LayerNorm and m.data_format == "channels_first":
                    m.__class__ = FastLayerNorm
        if self.channels_last:
            self.to(memory_format=torch.channels_last)
        log.info("model built: gen_bins=%d (start %d) anchors=%s aligned=%s ckpt=%s params=%.2fM",
                 self.hr_freq_bins, gen_start_bin, self.bw_anchor_bins, aligned_input,
                 grad_checkpoint, sum(p.numel() for p in self.parameters()) / 1e6)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def _pad_frames(self, x):
        pad_len = (self.strides - x.shape[-1] % self.strides) % self.strides
        if pad_len:
            x = F.pad(x, [0, pad_len, 0, 0], mode='reflect')
        return x, pad_len

    def _bins(self, cutoff_bins, y, B, device):
        if cutoff_bins is None:
            if y is None:
                raise ValueError("cutoff_bins is required when y is None")
            cutoff_bins = y.shape[2]
        cb = torch.as_tensor(cutoff_bins, dtype=torch.float32, device=device).reshape(-1)
        if cb.numel() == 1:
            cb = cb.expand(B)
        return cb.round().clamp(1, self.total_freq_bins)

    def forward(self, x, t, y, cutoff_bins=None):
        """
        x: noisy generated-region spec [B,2,hr_freq_bins,T]
        t: time [B] or [B,1]
        y: LQ spec [B,2,Fy,T] covering bins from 0 (Fy >= max cutoff bins); None = unconditional
        cutoff_bins: valid LQ bins per sample [B] (or scalar)
        """
        x, pad_len = self._pad_frames(x)
        if pad_len and y is not None:
            y = F.pad(y, [0, pad_len, 0, 0], mode='reflect')
        B, _, Fg, T = x.shape
        if Fg != self.hr_freq_bins:
            raise ValueError(f"x has {Fg} freq bins, model generates {self.hr_freq_bins}")
        cb = self._bins(cutoff_bins, y, B, x.device)

        pe_full = self.freq_pos_enc()
        pe_high = pe_full[self.gen_start_bin:]
        bw_emb = self.bw_embedder(cb)
        t_embed = self.time_embedder(t) + self.sr_projector(bw_emb)

        drop = None
        if y is not None:
            fc = int(min(y.shape[2], math.ceil(cb.max().item())))
            y_low = y[:, :, :fc]
            idx = torch.arange(fc, device=x.device)
            mask = (idx[None, :] < cb[:, None]).to(y_low.dtype)[:, None, :, None]
            y_real = self.conditioning_encoder(y_low, pe_full[:fc], bw_emb, mask)
            y_cond = y_real
            if self.training and self.cond_dropout_prob > 0:
                drop = torch.rand(B, device=x.device) < self.cond_dropout_prob
                uncond = self.uncond_emb.reshape(1, self.cond_dim, 1).expand(B, self.cond_dim, T)
                y_cond = torch.where(drop.reshape(B, 1, 1), uncond, y_real)
        else:
            y_cond = self.uncond_emb.reshape(1, self.cond_dim, 1).expand(B, self.cond_dim, T)

        gamma_high, beta_high = torch.chunk(self.film_generator(pe_high), 2, dim=-1)
        gamma_high = rearrange(gamma_high, 'f d -> 1 d f 1')
        beta_high = rearrange(beta_high, 'f d -> 1 d f 1')
        feats = [x, y_cond.unsqueeze(2) * gamma_high + beta_high]

        if self.aligned_input:
            if y is None:
                aligned = x.new_zeros(B, self.aligned_extra, Fg, T)
            else:
                ya = y[:, :, self.gen_start_bin:self.gen_start_bin + Fg]
                if ya.shape[2] < Fg:
                    ya = F.pad(ya, [0, 0, 0, Fg - ya.shape[2]])
                bin_idx = torch.arange(self.gen_start_bin, self.total_freq_bins, device=x.device)
                valid = (bin_idx[None, :] < cb[:, None]).to(ya.dtype)[:, None, :, None].expand(B, 1, Fg, T)
                aligned = torch.cat([ya, valid], dim=1)
                if drop is not None:
                    aligned = aligned * (~drop).to(aligned.dtype).reshape(B, 1, 1, 1)
            feats.append(aligned)

        x = self.init_conv(torch.cat(feats, dim=1))
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        skips = [x]
        ck = self.grad_checkpoint
        for enc in self.encoders:
            for blk in enc.blocks:
                x = _run(blk, ck, x, t_embed)
            x = enc.downsampler(x)
            skips.append(x)
        for blk in self.midcoder.blocks:
            x = _run(blk, ck, x, t_embed)
        for dec in self.decoders:
            skip = skips.pop()
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = x + skip
            x = dec.upsampler(x)
            for blk in dec.blocks:
                x = _run(blk, ck, x, t_embed)
        x = self.final_conv(x + skips.pop())
        return x[..., :-pad_len] if pad_len else x
