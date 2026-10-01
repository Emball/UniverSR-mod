import logging
import math
import os
import threading
import time
from collections import defaultdict

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, StepLR

log = logging.getLogger("universr.train")


def apply_optimizations(opt_cfg, repo_root):
    opt_cfg = opt_cfg or {}
    tf32 = bool(opt_cfg.get("tf32", False))
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = bool(opt_cfg.get("cudnn_benchmark", True))
    if opt_cfg.get("triton_cache", True):
        cache_dir = os.path.join(repo_root, ".triton_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = cache_dir
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    log.info("[optimizations] tf32=%s cudnn_benchmark=%s", tf32, torch.backends.cudnn.benchmark)
    start_ram_watchdog(float(opt_cfg.get("ram_limit_fraction", 0.90)))


def start_ram_watchdog(limit_fraction):
    try:
        import psutil
    except ImportError:
        log.warning("[watchdog] psutil not installed -- RAM watchdog disabled")
        return None
    total = psutil.virtual_memory().total
    threshold = total * limit_fraction
    log.info("[watchdog] exits cleanly above %.1f GB (%.0f%% of total)", threshold / 1024 ** 3, limit_fraction * 100)

    def watch():
        time.sleep(30)
        while True:
            used = psutil.virtual_memory().used
            if used >= threshold:
                log.error("[watchdog] SYSTEM RAM CRITICAL: %.1f GB used (threshold %.1f GB) -- exiting",
                          used / 1024 ** 3, threshold / 1024 ** 3)
                os._exit(1)
            time.sleep(2)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    return t


def resolve_precision(requested):
    if requested not in (None, "auto"):
        return str(requested)
    if not torch.cuda.is_available():
        return "32-true"
    # is_bf16_supported() reports True on Turing via emulation, which is slow; require native (Ampere+)
    native_bf16 = torch.cuda.get_device_capability()[0] >= 8
    return "bf16-mixed" if native_bf16 else "16-mixed"


class ComboOptimizer(Optimizer):
    # Presents several optimizers as one torch Optimizer so schedulers and Lightning accept it
    def __init__(self, opts):
        self.opts = [o for o in opts if o is not None]
        if not self.opts:
            raise ValueError("ComboOptimizer needs at least one optimizer")
        self.defaults = dict(self.opts[0].defaults)
        self.param_groups = [g for o in self.opts for g in o.param_groups]
        self.state = defaultdict(dict)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for o in self.opts:
            o.step()
        return loss

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"opts": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd):
        for o, s in zip(self.opts, sd["opts"]):
            o.load_state_dict(s)
        self.param_groups = [g for o in self.opts for g in o.param_groups]


def _gefen():
    import pkg_resources
    import packaging
    if not hasattr(pkg_resources, "packaging"):
        pkg_resources.packaging = packaging
    import gefen
    return gefen


def make_optimizer(params, cfg):
    params = list(params)
    kind = str(cfg.get("type", "adamw")).lower()
    lr = float(cfg.get("lr", 5e-5))
    wd = float(cfg.get("weight_decay", 0.01))
    betas = tuple(cfg.get("betas", [0.9, 0.99]))

    if kind in ("gefen", "gefen_muon"):
        try:
            g = _gefen()
            if kind == "gefen":
                log.info("[optimizer] Gefen lr=%g", lr)
                return g.Gefen(params, lr=lr, weight_decay=wd, betas=betas, fused=True)
            p2d = [p for p in params if p.ndim == 2]
            prest = [p for p in params if p.ndim != 2]
            o2d = g.GefenMuon(p2d, lr=lr, weight_decay=wd) if p2d else None
            orest = g.Gefen(prest, lr=lr, weight_decay=wd, betas=betas, fused=True) if prest else None
            log.info("[optimizer] GefenMuon(2D)=%d + Gefen(rest)=%d lr=%g", len(p2d), len(prest), lr)
            return ComboOptimizer([o2d, orest]) if o2d and orest else (o2d or orest)
        except ImportError as e:
            log.warning("[optimizer] gefen unavailable (%s) -- falling back to AdamW", e)

    if kind == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            log.info("[optimizer] AdamW8bit lr=%g", lr)
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=wd, betas=betas)
        except ImportError:
            log.warning("[optimizer] bitsandbytes not installed -- falling back to AdamW")

    if kind not in ("adamw", "gefen", "gefen_muon", "adamw_8bit"):
        log.warning("[optimizer] unknown type %r -- using AdamW", kind)
    log.info("[optimizer] AdamW lr=%g", lr)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)


def param_groups(named, lr, mult):
    """Split (name, param) pairs into AdamW groups; names starting with a key in `mult` get lr * factor."""
    named = list(named)
    for pre in mult:
        if not any(n.startswith(pre) for n, _ in named):
            raise ValueError(f"lr_mult prefix {pre!r} matches no trainable parameter")
    base, boosted = [], {}
    for n, p in named:
        f = next((float(v) for k, v in mult.items() if n.startswith(k)), None)
        if f is None:
            base.append(p)
        else:
            boosted.setdefault(f, []).append(p)
    groups = [{"params": base, "lr": lr}] if base else []
    for f, ps in boosted.items():
        log.info("[optimizer] lr x%g on %d tensors (%.2fM params)", f, len(ps), sum(p.numel() for p in ps) / 1e6)
        groups.append({"params": ps, "lr": lr * f})
    return groups


def make_scheduler(optimizer, cfg, max_steps):
    cfg = cfg or {}
    kind = str(cfg.get("type", "cosine")).lower()
    if kind == "steplr":
        return StepLR(optimizer, step_size=int(cfg.get("step_size", 1000)), gamma=float(cfg.get("gamma", 0.98)))
    warm = int(cfg.get("warmup_steps", 0))
    total = cfg.get("total_steps", "auto")
    total = int(max_steps) if total in (None, "auto") else int(total)
    min_ratio = float(cfg.get("min_lr_ratio", 0.0))
    if kind == "cosine" and total <= warm:
        raise ValueError(f"cosine scheduler needs total_steps ({total}) > warmup_steps ({warm}); set training.max_steps")

    def lam(step):
        if warm > 0 and step < warm:
            return (step + 1) / warm
        if kind == "constant":
            return 1.0
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    log.info("[scheduler] %s warmup=%d total=%d min_ratio=%g", kind, warm, total, min_ratio)
    return LambdaLR(optimizer, lam)


def freeze_by_prefix(model, prefixes):
    prefixes = [p for p in (prefixes or []) if p]
    if not prefixes:
        return 0, sum(1 for _ in model.parameters())
    names = [n for n, _ in model.named_parameters()]
    for pre in prefixes:
        if not any(n.startswith(pre) for n in names):
            raise ValueError(f"freeze prefix {pre!r} matches no parameter")
    frozen = 0
    for n, p in model.named_parameters():
        if any(n.startswith(pre) for pre in prefixes):
            p.requires_grad_(False)
            frozen += p.numel()
    total = sum(p.numel() for p in model.parameters())
    log.info("[freeze] %s -> %d / %d params frozen (%.1f%%)", prefixes, frozen, total, 100 * frozen / total)
    return frozen, total
