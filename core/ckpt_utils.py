import logging
import os
import re
from typing import Optional

import torch

log = logging.getLogger("universr.ckpt")

_FIELDS = ("step", "sisdr", "visqol", "hfnr")
_NUM = r"(-?\d+(?:\.\d+)?)"
_NAME_RE = re.compile(
    rf"^step={_NUM}(?:-sisdr={_NUM})?(?:-visqol={_NUM})?(?:-hfnr={_NUM})?(?:-v\d+)?\.ckpt$"
)


def parse_ckpt_name(name: str) -> Optional[dict]:
    m = _NAME_RE.match(os.path.basename(name))
    if not m:
        return None
    vals = [float(g) if g is not None else None for g in m.groups()]
    return dict(zip(_FIELDS, vals))


def rank_key(meta: dict):
    # visqol primary, sisdr tiebreak, lower hfnr final tiebreak; missing or unavailable (-1) visqol ranks last
    visqol = meta.get("visqol")
    visqol = float("-inf") if visqol is None or visqol < 0 else visqol
    sisdr = meta.get("sisdr")
    sisdr = float("-inf") if sisdr is None else sisdr
    hfnr = meta.get("hfnr")
    hfnr = float("inf") if hfnr is None else hfnr
    return (visqol, sisdr, -hfnr, meta.get("step") or 0.0)


def list_checkpoints(ckpt_dir: str):
    out = []
    if not os.path.isdir(ckpt_dir):
        return out
    for f in os.listdir(ckpt_dir):
        meta = parse_ckpt_name(f)
        if meta is not None:
            meta["path"] = os.path.join(ckpt_dir, f)
            out.append(meta)
    return out


def best_checkpoint(ckpt_dir: str) -> Optional[str]:
    items = list_checkpoints(ckpt_dir)
    if not items:
        return None
    return max(items, key=rank_key)["path"]


def latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    cands = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".ckpt") and f != "last.ckpt"]
    return max(cands, key=os.path.getmtime) if cands else None


def load_ckpt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        log.info("weights_only load failed, retrying with weights_only=False: %s", path)
        return torch.load(path, map_location="cpu", weights_only=False)


def model_state_from_ckpt(ckpt: dict, prefer_ema: bool = True) -> dict:
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state = {k[len("audio_model."):]: v for k, v in sd.items() if k.startswith("audio_model.")}
    if not state:
        state = dict(sd)
    ema = ckpt.get("ema") if prefer_ema else None
    if ema:
        n = 0
        for k, v in ema.items():
            if k in state and state[k].shape == v.shape:
                state[k] = v.to(state[k].dtype)
                n += 1
        log.info("applied EMA weights to %d tensors", n)
    return state


def export_model(ckpt_path: str, out_path: str, model, model_cfg: dict, transform_cfg: dict) -> None:
    ckpt = load_ckpt(ckpt_path)
    state = model_state_from_ckpt(ckpt)
    bundle = {
        "model_state_dict": state,
        "model_cfg": dict(model_cfg),
        "transform_cfg": {**dict(transform_cfg), **({"alpha": float(ckpt["alpha"])} if ckpt.get("alpha") is not None else {})},
        "gen_start_bin": int(model.gen_start_bin),
        "aligned_input": bool(model.aligned_input),
        "bw_anchor_bins": [int(b) for b in model.bw_anchor_bins],
        "source_ckpt": os.path.basename(ckpt_path),
    }
    torch.save(bundle, out_path)
    log.info("exported %s -> %s", os.path.basename(ckpt_path), out_path)
