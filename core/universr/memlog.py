import logging
import time

import torch

log = logging.getLogger(__name__)
_GB = 1024 ** 3
_last = {"t": None}


def _rss_free():
    try:
        import psutil
        return psutil.Process().memory_info().rss / _GB, psutil.virtual_memory().available / _GB
    except Exception:
        return float("nan"), float("nan")


def snap(tag, reset_peak=False):
    """One line: torch allocated/reserved/peak, driver-level used VRAM, allocator retries, process RSS, free RAM."""
    if not torch.cuda.is_available():
        return
    try:
        free, total = torch.cuda.mem_get_info()
        st = torch.cuda.memory_stats()
        rss, sysfree = _rss_free()
        now = time.time()
        dt = "" if _last["t"] is None else f" +{now - _last['t']:.1f}s"
        _last["t"] = now
        log.info(
            "[mem] %-30s alloc=%.2f reserved=%.2f peak=%.2f | driver_used=%.2f/%.2f GB | "
            "alloc_retries=%d ooms=%d frag_blocks=%d | rss=%.2f sys_free=%.2f GB%s",
            tag, torch.cuda.memory_allocated() / _GB, torch.cuda.memory_reserved() / _GB,
            torch.cuda.max_memory_allocated() / _GB, (total - free) / _GB, total / _GB,
            st.get("num_alloc_retries", 0), st.get("num_ooms", 0), st.get("inactive_split.all.current", 0),
            rss, sysfree, dt)
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        log.warning("[mem] snapshot failed: %s", e)
