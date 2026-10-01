import logging
import os
import sys

import torch
from omegaconf import OmegaConf

CORE = os.path.dirname(os.path.abspath(__file__))
if CORE not in sys.path:
    sys.path.insert(0, CORE)
REPO_ROOT = os.path.dirname(CORE)

from audio_io import decode_to_wav_cache, read_wav, resample  # noqa: E402
from ckpt_utils import list_checkpoints, load_ckpt, model_state_from_ckpt, rank_key  # noqa: E402
from cutoff import resolve_cutoff_hz  # noqa: E402
from data_prep import DECODE_DIR  # noqa: E402
from ensemble import merge_long  # noqa: E402
from universr.models.loader import build_model, load_state_dict_into, read_pretrained_config  # noqa: E402
from universr.sampling import make_transform, restore_long  # noqa: E402

log = logging.getLogger("universr.restorer")

MODELS_DIR = os.path.join(REPO_ROOT, "models")
PRETRAINED_NAMES = ("pytorch_model.bin", "universr.bin", "universr.pth", "universr.ckpt")
PRETRAINED_CONFIG = os.path.join(REPO_ROOT, "configs", "universr_pretrained.yaml")
HF_REPO = "woongzip1/universr-audio"


def find_pretrained():
    for name in PRETRAINED_NAMES:
        cand = os.path.join(MODELS_DIR, name)
        if os.path.isfile(cand):
            return cand
    try:
        from huggingface_hub import hf_hub_download
        log.info("no pretrained weights in models/ -- downloading %s", HF_REPO)
        return hf_hub_download(repo_id=HF_REPO, filename="pytorch_model.bin")
    except Exception as e:
        raise FileNotFoundError(f"No pretrained weights found. Place pytorch_model.bin from {HF_REPO} "
                                f"in models/ ({e})") from e


def run_dir_for(cfg, conf_dir):
    exp = cfg.get("exp") or {}
    base = str(exp.get("dir") or os.path.join(REPO_ROOT, "runs"))
    if not os.path.isabs(base) and not os.path.isdir(base):
        base = os.path.join(REPO_ROOT, base)
    name = str(exp.get("name") or os.path.splitext(os.path.basename(conf_dir))[0])
    return os.path.join(base, name)


def find_best_weights(cfg, conf_dir):
    """Best ranked checkpoint across every run folder of the experiment."""
    run_dir = run_dir_for(cfg, conf_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"No run directory at {run_dir!r}")
    items = []
    for sub in os.listdir(run_dir):
        items += list_checkpoints(os.path.join(run_dir, sub, "checkpoints"))
    if items:
        best = max(items, key=rank_key)["path"]
        log.info("auto-selected checkpoint: %s", os.path.relpath(best))
        return best
    raise FileNotFoundError(f"No ranked checkpoints under {run_dir!r}")


def pick_amp_dtype(precision, device):
    p = str(precision).lower()
    if device.type != "cuda" or p in ("fp32", "32", "32-true"):
        return None
    if p in ("fp16", "16", "16-mixed"):
        return torch.float16
    if p in ("bf16", "bf16-mixed"):
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


class Restorer:
    def __init__(self, model, transform, sr, device, amp_dtype, steps, guidance, keep_lq, source):
        self.model, self.transform, self.sr = model, transform, int(sr)
        self.device, self.amp_dtype = device, amp_dtype
        self.steps, self.guidance, self.keep_lq = int(steps), guidance, keep_lq
        self.source = source
        self.lsd_cutoff_hz = None


def load_restorer(weights=None, conf_dir=None, device="auto", precision="auto", prefer_ema=True):
    cfg = OmegaConf.load(conf_dir) if conf_dir else None
    if weights in (None, ""):
        weights = find_best_weights(cfg, conf_dir) if cfg is not None else find_pretrained()
    elif str(weights).lower() == "pretrained":
        weights = find_pretrained()
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"weights not found: {weights}")
    log.info("loading %s", weights)

    ext = os.path.splitext(weights)[1].lower()
    ck = load_ckpt(weights) if ext in (".ckpt", ".pth") else None
    if ck is not None and "model_state_dict" in ck and "model_cfg" in ck:
        mc, tcfg = dict(ck["model_cfg"]), dict(ck["transform_cfg"])
        model = build_model(mc, gen_start_bin=ck["gen_start_bin"], aligned_input=ck["aligned_input"],
                            bw_anchor_bins=ck["bw_anchor_bins"])
        load_state_dict_into(model, ck["model_state_dict"])
    else:
        if cfg is not None:
            mc = OmegaConf.to_container(cfg.model, resolve=True)
            tcfg = OmegaConf.to_container(cfg.transform, resolve=True)
        elif ext == ".ckpt":
            raise ValueError("a Lightning .ckpt needs --conf_dir (model settings are not stored in it); "
                             "use the exported best_model.pth to run without a config")
        else:
            sib = os.path.join(os.path.dirname(os.path.abspath(weights)), "config.yaml")
            base = read_pretrained_config(sib if os.path.isfile(sib) else PRETRAINED_CONFIG)
            mc, tcfg = base["model"], base["transform"]
        model = build_model(mc, gen_start_bin=mc.get("gen_start_bin"), aligned_input=bool(mc.get("aligned_input", False)),
                            bw_anchor_bins=mc.get("bw_anchor_bins"))
        if ck is not None:
            load_state_dict_into(model, model_state_from_ckpt(ck, prefer_ema=prefer_ema))
        else:
            raw = torch.load(weights, map_location="cpu", weights_only=False)
            load_state_dict_into(model, raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw)

    sc = (OmegaConf.to_container(cfg.get("system", {}), resolve=True) if cfg is not None else {}) or {}
    keep = sc.get("keep_lq_below_cutoff", "auto")
    keep = (not model.aligned_input) if keep in ("auto", None) else bool(keep)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    model = model.to(dev).eval()
    if ck is not None and ck.get("alpha") is not None and ext == ".ckpt":
        tcfg = {**tcfg, "alpha": float(ck["alpha"])}
    transform = make_transform(tcfg)
    if hasattr(transform, "to"):
        transform = transform.to(dev)
    rs = Restorer(model, transform, tcfg["sampling_rate"], dev, pick_amp_dtype(precision, dev),
                  sc.get("val_ode_steps", 4), sc.get("val_guidance", 1.5), keep, os.path.basename(weights))
    rs.lsd_cutoff_hz = sc.get("lsd_cutoff_hz")
    rs.band_gain_db = [list(map(float, b)) for b in (sc.get("val_band_gain_db") or [])] or None
    log.info("ready: sr=%d device=%s amp=%s steps=%d guidance=%s aligned_input=%s keep_lq_below_cutoff=%s",
             rs.sr, dev, rs.amp_dtype, rs.steps, rs.guidance, model.aligned_input, keep)
    return rs


def load_input(path):
    wav_path = decode_to_wav_cache(path, DECODE_DIR)
    wav, sr = read_wav(wav_path)
    if wav.shape[-1] == 0:
        raise ValueError(f"audio file has no samples: {path}")
    if not torch.isfinite(wav).all():
        raise ValueError(f"audio file contains NaN or Inf: {path}")
    return wav[:2], sr


def resolve_cutoff(spec, mono_np, sr):
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return float(min(spec, sr / 2))
    if isinstance(spec, str) and spec not in ("auto", ""):
        return float(min(float(spec), sr / 2))
    return resolve_cutoff_hz("auto", mono_np, sr)


def _run_channels(rs, xs, cutoff_hz, chunk_sec, overlap_sec, steps, guidance, seed, shared_noise, progress, tag,
                  band_gain_db=None):
    outs = []
    for c in range(xs.shape[0]):
        def cb(done, total, c=c):
            if progress:
                progress(f"{tag} ch{c + 1}/{xs.shape[0]}", done, total)
        out = restore_long(
            rs.model, rs.transform, xs[c:c + 1].to(rs.device), cutoff_hz, rs.sr,
            chunk_sec=chunk_sec, overlap_sec=overlap_sec, progress=cb,
            steps=rs.steps if steps is None else steps, guidance=rs.guidance if guidance is None else guidance,
            keep_lq_below_cutoff=rs.keep_lq, seed=seed if shared_noise else seed + c, amp_dtype=rs.amp_dtype,
            band_gain_db=getattr(rs, "band_gain_db", None) if band_gain_db is None else band_gain_db)
        outs.append(out.cpu())
    return torch.cat(outs, dim=0)


def restore_audio(rs, wav, sr_in, cutoff="auto", chunk_sec=6.0, overlap_sec=0.5, steps=None, guidance=None,
                  seed=1234, shared_noise=False, bands=None, aux=None, aux_bands=None, band_gain_db=None,
                  target_peak_dbfs=-3.0, match_input_sr=False, progress=None):
    """[C,N] at sr_in -> ([C,N'], sr_out). Channels are restored independently, as in Apollo-mod."""
    x = resample(wav[:2].float(), sr_in, rs.sr)
    peak = x.abs().max().item()
    out_sr = sr_in if match_input_sr else rs.sr
    if peak < 1e-4:
        log.warning("input is silent -- returning it unchanged")
        return (wav if match_input_sr else x), out_sr
    scale = (10 ** (target_peak_dbfs / 20.0)) / peak
    xs = x * scale
    cutoff_hz = resolve_cutoff(cutoff, x.mean(0).numpy(), rs.sr)
    log.info("cutoff %.0f Hz, input peak %.2f dBFS -> %.1f dBFS for the model", cutoff_hz,
             20 * torch.log10(torch.tensor(peak)).item(), target_peak_dbfs)

    enhanced = _run_channels(rs, xs, cutoff_hz, chunk_sec, overlap_sec, steps, guidance, seed, shared_noise,
                             progress, "restoring", band_gain_db) / scale
    if bands:
        enhanced = merge_long(x, enhanced, rs.sr, bands)
    if aux is not None:
        if aux.sr != rs.sr:
            raise ValueError(f"aux model sample rate {aux.sr} differs from primary {rs.sr}")
        aux_out = _run_channels(aux, xs, cutoff_hz, chunk_sec, overlap_sec, steps, guidance, seed, shared_noise,
                                progress, "aux", band_gain_db) / scale
        ab = aux_bands or bands
        if ab:
            enhanced = merge_long(aux_out, enhanced, rs.sr, ab)
    if match_input_sr:
        enhanced = resample(enhanced, rs.sr, sr_in)
    return enhanced, out_sr
