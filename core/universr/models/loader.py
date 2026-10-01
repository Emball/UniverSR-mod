import logging

import torch
import yaml

from universr.models.unet_mod import (
    PRETRAINED_ANCHOR_BINS,
    BandwidthEmbedding,
    ConvNeXtUNetCondMod,
)

log = logging.getLogger("universr.loader")

DEFAULT_ANCHOR_BINS = (80, 128, 170, 256, 341, 405, 470, 512)


def default_anchor_bins(total):
    return tuple(a for a in DEFAULT_ANCHOR_BINS[:-1] if a < total) + (int(total),)


def read_pretrained_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_cfg, gen_start_bin=None, aligned_input=False,
                bw_anchor_bins=None, grad_checkpoint=False):
    cfg = dict(model_cfg)
    cfg.pop("sr_to_lr_bins", None)
    hr = cfg.pop("hr_freq_bins", None)
    total = cfg.get("total_freq_bins", 512)
    if gen_start_bin is None:
        gen_start_bin = total - hr if hr is not None else cfg.pop("gen_start_bin", 80)
    cfg.pop("gen_start_bin", None)
    cfg.pop("bw_anchor_bins", None)
    cfg.pop("aligned_input", None)
    cfg.pop("grad_checkpoint", None)
    return ConvNeXtUNetCondMod(
        gen_start_bin=gen_start_bin,
        aligned_input=aligned_input,
        bw_anchor_bins=tuple(bw_anchor_bins or default_anchor_bins(total)),
        grad_checkpoint=grad_checkpoint,
        **cfg,
    )


@torch.no_grad()
def _remap_bandwidth(old, model):
    if old.shape[0] != len(PRETRAINED_ANCHOR_BINS):
        raise ValueError(f"unexpected sr_embedder rows: {tuple(old.shape)}")
    tmp = BandwidthEmbedding(PRETRAINED_ANCHOR_BINS, old.shape[1])
    tmp.weight.copy_(old)
    anchors = torch.tensor(model.bw_anchor_bins, dtype=torch.float32)
    return tmp(anchors)


def reconcile_state_dict(sd, model):
    sd = dict(sd)
    msd = model.state_dict()
    report = {"padded": [], "remapped": [], "skipped": [], "unexpected": [], "missing": []}

    if "sr_embedder.weight" in sd:
        sd["bw_embedder.weight"] = _remap_bandwidth(sd.pop("sr_embedder.weight"), model)
        report["remapped"].append("sr_embedder.weight -> bw_embedder.weight")

    out = {}
    for k, v in sd.items():
        if k not in msd:
            report["unexpected"].append(k)
            continue
        tgt = msd[k].shape
        if v.shape == tgt:
            out[k] = v
        elif k == "freq_pos_enc.pe" and v.dim() == 2 and v.shape[1] == tgt[1] and v.shape[0] > tgt[0]:
            out[k] = v[:tgt[0]]
            report["remapped"].append(f"{k}: sliced {tuple(v.shape)} -> {tuple(tgt)}")
        elif (v.dim() == 4 and v.shape[0] == tgt[0] and v.shape[2:] == tgt[2:]
              and v.shape[1] < tgt[1] and k == "init_conv.0.weight"):
            pad = torch.zeros(tgt[0], tgt[1] - v.shape[1], *tgt[2:], dtype=v.dtype)
            out[k] = torch.cat([v, pad], dim=1)
            report["padded"].append(f"{k}: {tuple(v.shape)} -> {tuple(tgt)}")
        else:
            report["skipped"].append(f"{k}: {tuple(v.shape)} vs {tuple(tgt)}")

    report["missing"] = [k for k in msd if k not in out]
    for key, items in report.items():
        for item in items:
            log.info("load %s: %s", key, item)
    return out, report


def load_state_dict_into(model, sd):
    new_sd, report = reconcile_state_dict(sd, model)
    model.load_state_dict(new_sd, strict=False)
    if report["skipped"] or report["missing"]:
        log.warning("partial load: %d skipped, %d missing", len(report["skipped"]), len(report["missing"]))
    else:
        log.info("loaded %d tensors (%d padded, %d remapped)",
                 len(new_sd), len(report["padded"]), len(report["remapped"]))
    return report


def load_pretrained(model, path):
    log.info("loading pretrained weights: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    return load_state_dict_into(model, sd)
