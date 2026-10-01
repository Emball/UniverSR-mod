"""
evaluate.py -- offline checkpoint evaluator for UniverSR-mod.

Launched from the TUI, or standalone:
    python core/evaluate.py --conf_dir configs/universr_stfl2.yaml [--visqol] [--baseline]

Every checkpoint is run on the same fixed set of validation clips (equal share per song), so the
numbers are comparable across checkpoints, unlike the rotating-window values in checkpoint filenames.
Metrics: visqol (primary, optional), hfnr (lower is better), sisdr, lsd_high.
Results are cached in <ckpt_dir>/.eval_cache.json, keyed by file and evaluation settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf, open_dict

CORE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CORE)
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from data_prep import prepare_data  # noqa: E402
from paired_datamodule import ChunkedPairDataset  # noqa: E402
from restorer import load_restorer  # noqa: E402
from universr import metrics as M  # noqa: E402
from universr.sampling import restore_long  # noqa: E402
from universr.system import _song_key  # noqa: E402

log = logging.getLogger("universr.evaluate")

_EVAL_WEIGHTS = {"visqol": 0.60, "hfnr": 0.25, "sisdr": 0.15}
_HIGHER_IS_BETTER = {"visqol", "sisdr"}
_METRIC_KEYS = ("visqol", "lsd_high", "hfnr", "sisdr")
BASELINE_NAME = "pretrained (zero-shot)"


def _visqol_available() -> bool:
    return M._get_visqol() is not None


def _pick_clips(dataset, limit):
    by_song = defaultdict(list)
    for i, (lq_path, _) in enumerate(dataset.pairs):
        by_song[_song_key(lq_path)].append(i)
    if not limit or limit <= 0 or limit >= len(dataset):
        return sorted(i for v in by_song.values() for i in v), by_song
    songs = sorted(by_song, key=lambda k: -len(by_song[k]))
    base, rem = divmod(int(limit), len(songs))
    picked = []
    for n, s in enumerate(songs):
        picked += by_song[s][:base + (1 if n < rem else 0)]
    return sorted(picked), by_song


def evaluate_restorer(rs, dataset, indices, run_visqol, seed, print_fn=print):
    rows = []
    with torch.no_grad():
        for idx in indices:
            hq, lq, _, song, cut = dataset[idx]
            est = restore_long(rs.model, rs.transform, lq.to(rs.device), float(cut), rs.sr, chunk_sec=None,
                               steps=rs.steps, guidance=rs.guidance, keep_lq_below_cutoff=rs.keep_lq,
                               seed=seed + idx, amp_dtype=rs.amp_dtype).cpu().clamp(-1.0, 1.0)
            row = {"sisdr": M.sisdr(est, hq), "hfnr": M.hfnr(est, hq, rs.sr),
                   "lsd_high": M.lsd(est, hq, rs.lsd_cutoff_hz or float(cut), rs.sr)[1]}
            if run_visqol:
                v = M.visqol(est, hq, rs.sr)
                if v is not None:
                    row["visqol"] = v
            rows.append(row)
    out = {"n": len(rows)}
    for k in _METRIC_KEYS:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def _rank_results(results):
    ranges = {}
    for key in _EVAL_WEIGHTS:
        present = [m[key] for _, _, m in results if key in m]
        if present:
            ranges[key] = (min(present), max(present))

    def composite(m):
        score = total = 0.0
        for key, w in _EVAL_WEIGHTS.items():
            if key not in m or key not in ranges:
                continue
            lo, hi = ranges[key]
            n = 0.0 if hi == lo else (m[key] - lo) / (hi - lo)
            if key in _HIGHER_IS_BETTER:
                n = 1.0 - n
            score += w * n
            total += w
        return score / total if total else float("inf")

    scored = [(f, p, dict(m, composite=composite(m))) for f, p, m in results]
    scored.sort(key=lambda x: x[2]["composite"])
    for rank, (_, _, m) in enumerate(scored, 1):
        m["rank"] = rank
    return scored


def _load_cache(ckpt_dir):
    try:
        return json.loads(Path(os.path.join(ckpt_dir, ".eval_cache.json")).read_text())
    except Exception:
        return {}


def _save_cache(ckpt_dir, cache):
    try:
        Path(os.path.join(ckpt_dir, ".eval_cache.json")).write_text(json.dumps(cache, indent=2))
    except Exception as ex:
        log.warning("cache write failed: %s", ex)


def _settings_key(path, rs_probe, cfg, limit, seed, eval_dir):
    st = os.stat(path) if path and os.path.isfile(path) else None
    payload = {"size": st.st_size if st else 0, "mtime": int(st.st_mtime) if st else 0, "limit": limit, "seed": seed,
               "steps": rs_probe["steps"], "guidance": rs_probe["guidance"], "keep": rs_probe["keep"],
               "val": os.path.basename(os.path.dirname(os.path.normpath(eval_dir))),
               "cutoff": OmegaConf.to_container(cfg.datas, resolve=True).get("cutoff_hz", "auto")}
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def run_evaluation(conf_dir: str, ckpt_dir: Optional[str] = None, limit: Optional[int] = None,
                   run_visqol: bool = False, pattern: Optional[str] = None, print_fn=print,
                   baseline: bool = False, device: str = "auto") -> list:
    cfg = OmegaConf.load(conf_dir)
    with open_dict(cfg):
        if not cfg.get("exp"):
            cfg.exp = {}
        if not cfg.exp.get("dir"):
            cfg.exp.dir = os.path.join(REPO_ROOT, "runs")
        if not cfg.exp.get("name"):
            cfg.exp.name = os.path.splitext(os.path.basename(conf_dir))[0]
    if limit is None:
        limit = int(cfg.get("training", {}).get("eval_clips", 12))
    seed = int((cfg.get("system", {}) or {}).get("val_seed", 1234))

    if ckpt_dir is None:
        run_root = os.path.join(cfg.exp.dir, cfg.exp.name)
        runs = sorted((d for d in os.listdir(run_root) if os.path.isdir(os.path.join(run_root, d, "checkpoints")))
                      if os.path.isdir(run_root) else [], reverse=True)
        if not runs:
            raise RuntimeError(f"No runs with checkpoints in {run_root}")
        ckpt_dir = os.path.join(run_root, runs[0], "checkpoints")
        print_fn(f"[eval] Using run: {runs[0]}")
    if not os.path.isdir(ckpt_dir):
        raise RuntimeError(f"Checkpoint dir not found: {ckpt_dir}")
    ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt") and f != "last.ckpt"
                   and (pattern is None or pattern in f))
    if not ckpts and not baseline:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")

    prepare_data(cfg)
    dc = cfg.datas
    dataset = ChunkedPairDataset(dc.eval_dir, int(dc.sr), None, "Validation", None, dc.get("cutoff_hz", "auto"),
                                 None, dc.get("val_peak_dbfs", -3.0), float(dc.get("cache_peak_dbfs", -1.0)))
    if len(dataset) == 0:
        raise RuntimeError(f"No validation clips in {dc.eval_dir}")
    indices, by_song = _pick_clips(dataset, limit)
    print_fn(f"[eval] {len(indices)} clips across {len(by_song)} songs")

    if run_visqol and not _visqol_available():
        print_fn('[eval] WARNING: visqol-python not installed -- skipping ViSQOL. Run: uv pip install "visqol-python[all]"')
        run_visqol = False

    cache = _load_cache(ckpt_dir)
    results = []
    jobs = [(f, os.path.join(ckpt_dir, f)) for f in ckpts]
    if baseline:
        jobs.insert(0, (BASELINE_NAME, "pretrained"))
    print_fn(f"[eval] {len(jobs)} model(s)\n")

    for name, path in jobs:
        is_base = path == "pretrained"
        try:
            rs = load_restorer(path, conf_dir, device)
            if is_base:
                rs.keep_lq = True
            key = _settings_key(None if is_base else path, {"steps": rs.steps, "guidance": rs.guidance,
                                                            "keep": rs.keep_lq}, cfg, limit, seed, dc.eval_dir)
            hit = cache.get(name)
            if hit and hit.get("key") == key and (not run_visqol or "visqol" in hit["metrics"]):
                print_fn(f"  Cached:    {name}")
                results.append((name, path, hit["metrics"]))
            else:
                print_fn(f"  Evaluating: {name} ...")
                m = evaluate_restorer(rs, dataset, indices, run_visqol, seed, print_fn)
                if hit and hit.get("key") == key:
                    m = {**hit["metrics"], **m}
                cache[name] = {"key": key, "metrics": m}
                _save_cache(ckpt_dir, cache)
                print_fn("    " + "  ".join(f"{k}={m[k]:.4f}" for k in _METRIC_KEYS if k in m))
                results.append((name, path, m))
            del rs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as ex:
            print_fn(f"ERROR: {name}: {ex}")
            log.exception("evaluation failed for %s", name)

    return _rank_results(results) if results else []


def print_results_table(results: list, print_fn=print) -> None:
    cols = [c for c in _METRIC_KEYS if any(c in m for _, _, m in results)]
    print_fn("\n" + "=" * 96)
    print_fn(f"{'Rank':<5}" + "".join(f" {c.upper():>9}" for c in cols) + "  Model")
    print_fn("-" * 96)
    for fname, _, m in results:
        row = f"  {m.get('rank', '?'):<3}"
        for c in cols:
            row += f" {m[c]:>9.4f}" if c in m else f" {'--':>9}"
        print_fn(row + f"  {fname}")
    print_fn("=" * 96)
    if results:
        best_name, _, best = results[0]
        print_fn(f"\nBest: {best_name}")
        print_fn("  " + "  ".join(f"{k}={best[k]:.4f}" for k in _METRIC_KEYS if k in best))
        print_fn("  (SI-SDR is expected to trail for a generative model; ViSQOL and listening are the fairer judges)")


def screen_evaluate(state: dict, console, _pick, _run_with_live_output, ROOT: Path) -> None:
    """Entry point called from tui.py."""
    configs_dir = ROOT / "configs"
    configs = sorted(configs_dir.glob("*.yaml")) if configs_dir.exists() else []
    if not configs:
        console.clear()
        console.print("[red]No configs found in configs/[/]")
        console.input("Press Enter to return.")
        return

    last_cfg = state.get("evaluate", {}).get("last_config", "")
    start = next((i for i, c in enumerate(configs) if c.name == last_cfg), 0)
    idx = _pick("Evaluate -- select config", [c.stem for c in configs], hint="Enter=select  Esc=back", start=start)
    if idx is None:
        return
    cfg_path = configs[idx]
    state.setdefault("evaluate", {})["last_config"] = cfg_path.name

    has_visqol = _visqol_available()
    visqol_label = "Run ViSQOL (slow -- perceptual score)" if has_visqol else \
        'Run ViSQOL  [not installed -- uv pip install "visqol-python[all]"]'
    midx = _pick("Evaluate -- mode", ["Fast  (hfnr, SI-SDR, LSD only)", visqol_label], hint="Enter=select  Esc=back")
    if midx is None:
        return
    run_visqol = midx == 1 and has_visqol
    bidx = _pick("Evaluate -- baseline", ["Checkpoints only", "Checkpoints + zero-shot pretrained baseline"],
                 hint="Enter=select  Esc=back")
    if bidx is None:
        return

    console.clear()
    from rich.panel import Panel
    console.print(Panel(f"[bold cyan]Evaluating: {cfg_path.stem}[/]\n"
                        f"[dim]{'ViSQOL enabled' if run_visqol else 'Fast mode (no ViSQOL)'}"
                        f"{'  +  zero-shot baseline' if bidx == 1 else ''}[/]", border_style="cyan", padding=(0, 2)))

    def _print(*args, **kwargs):
        console.print(" ".join(str(a) for a in args))

    try:
        results = run_evaluation(conf_dir=str(cfg_path), run_visqol=run_visqol, baseline=bidx == 1, print_fn=_print)
        if results:
            print_results_table(results, print_fn=_print)
        else:
            console.print("[yellow]No results.[/]")
    except Exception as ex:
        console.print(f"[red]Error: {ex}[/]")
    console.input("\n[dim]Press Enter to return to menu[/]")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    ap = argparse.ArgumentParser(description="Evaluate UniverSR-mod checkpoints on a fixed clip set")
    ap.add_argument("--conf_dir", required=True, help="path to yaml config")
    ap.add_argument("--ckpt_dir", default=None, help="checkpoint folder (newest run if omitted)")
    ap.add_argument("--limit", type=int, default=None, help="total val clips across songs (default: training.eval_clips, else 12)")
    ap.add_argument("--visqol", action="store_true", help="run ViSQOL (requires visqol-python)")
    ap.add_argument("--baseline", action="store_true", help="also evaluate the zero-shot pretrained model (real bins kept below cutoff)")
    ap.add_argument("--pattern", default=None, help="only evaluate checkpoints containing this substring")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    try:
        results = run_evaluation(a.conf_dir, a.ckpt_dir, a.limit, a.visqol, a.pattern, baseline=a.baseline, device=a.device)
    except RuntimeError as ex:
        print(f"[error] {ex}")
        sys.exit(1)
    if not results:
        print("\n[eval] No results.")
        sys.exit(1)
    print_results_table(results)


if __name__ == "__main__":
    main()
