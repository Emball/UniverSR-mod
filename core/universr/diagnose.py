import logging

import torch

log = logging.getLogger("universr.diagnose")


def describe(name, t):
    t = t.detach().float()
    fin = torch.isfinite(t)
    if bool(fin.all()):
        return f"{name}: ok  shape={tuple(t.shape)} absmax={t.abs().max().item():.4g}"
    bad = int((~fin).sum())
    return f"{name}: NON-FINITE {bad}/{t.numel()}  shape={tuple(t.shape)}"


@torch.no_grad()
def trace(model, *args, autocast=True, top=5):
    """One forward with hooks on leaf modules. Returns (first non-finite module or None, largest activations)."""
    stats, first, hooks = [], [], []

    def make(name):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) and out else out
            if not torch.is_tensor(t) or not t.is_floating_point():
                return
            fin = torch.isfinite(t)
            ok = bool(fin.all())
            amax = float(t[fin].abs().max()) if bool(fin.any()) else float("nan")
            stats.append((amax, name, str(t.dtype).replace("torch.", "")))
            if not ok and not first:
                first.append(name)
        return hook

    for name, m in model.named_modules():
        if name and not any(True for _ in m.children()):
            hooks.append(m.register_forward_hook(make(name)))
    dev = args[0].device.type
    try:
        with torch.autocast(device_type=dev, enabled=autocast):
            model(*args)
    except Exception as e:
        log.warning("[nan-diag] trace forward raised: %s", e)
    finally:
        for h in hooks:
            h.remove()
    stats.sort(key=lambda s: -(s[0] if s[0] == s[0] else float("inf")))
    return (first[0] if first else None), stats[:top]


def run(model, xt, t, Y, cutoff_bins, extra_inputs=None):
    """Log which tensor or module first goes non-finite, and whether fp32 is clean."""
    for name, tensor in (extra_inputs or {}).items():
        log.warning("[nan-diag] %s", describe(name, tensor))
    log.warning("[nan-diag] %s", describe("xt", xt))
    log.warning("[nan-diag] %s", describe("Y", Y))
    log.warning("[nan-diag] %s", describe("cutoff_bins", torch.as_tensor(cutoff_bins, dtype=torch.float32)))
    was_training = model.training
    for label, ac in (("autocast", True), ("fp32", False)):
        first, big = trace(model, xt, t, Y, cutoff_bins, autocast=ac)
        if first is None:
            log.warning("[nan-diag] %s: all module outputs finite", label)
        else:
            log.warning("[nan-diag] %s: FIRST non-finite module: %s", label, first)
        log.warning("[nan-diag] %s largest activations: %s", label,
                    "  ".join(f"{n}={v:.4g}({d})" for v, n, d in big))
    model.train(was_training)
