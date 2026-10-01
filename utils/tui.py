"""
Apollo TUI -- keyboard-navigated launcher for train / inference / utilities.

Navigation: arrow keys or j/k, Enter to select, Escape or q to go back.
During training: output streams live; press Ctrl+C to stop and return to menu.
"""
from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import threading
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"

if not IS_WINDOWS:
    import termios
    import tty

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.resolve()  # repo root (parent of utils/)
sys.path.insert(0, str(ROOT / "core"))          # for lazy imports of evaluate, degrade_audio
sys.path.insert(0, str(ROOT / "utils"))         # for degrade_audio co-located in utils/
CONFIGS_DIR = ROOT / "configs"
DEV_CONFIGS_DIR = ROOT / "dev"
RUNS_DIR = ROOT / "runs"
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"
STATE_FILE = ROOT / "usr" / ".tui_state.json"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".aiff", ".aif"}

console = Console()

# ---------------------------------------------------------------------------
# Legacy checkpoint filename migration
# ---------------------------------------------------------------------------

def _migrate_checkpoint_name(path: Path) -> Path:
    """
    Rename a single checkpoint file from any legacy naming scheme to the current one.
    Returns the (possibly new) path. No-ops if the name is already current.

    Legacy → current:
      val_loss=    → sisdr=
      val_visqol=  → visqol=
      val_sfr=     → hfnr=
      val_hfnr=    → hfnr=
      -sdr=<val>   → stripped  (bare sdr, not sisdr)
    """
    import re as _re
    stem = path.stem          # excludes .ckpt
    new  = stem

    new = new.replace("val_loss=",   "sisdr=")
    new = new.replace("val_visqol=", "visqol=")
    new = new.replace("val_sfr=",    "hfnr=")
    new = new.replace("val_hfnr=",   "hfnr=")
    # Strip -val_sdr=<value> and bare -sdr=<value> segments (not sisdr=)
    new = _re.sub(r"-val_sdr=[\d.]+", "", new)
    new = _re.sub(r"-(?<!si)sdr=[\d.]+", "", new)

    if new == stem:
        return path  # nothing changed

    new_path = path.with_name(new + ".ckpt")
    if new_path.exists():
        # Target already exists — stale duplicate from partial migration; delete the old file
        try:
            path.unlink()
        except Exception:
            pass
        return new_path
    try:
        path.rename(new_path)
    except Exception as exc:
        print(f"[migrate] could not rename {path.name}: {exc}")
        return path
    return new_path


def migrate_checkpoints(runs_dir: Path) -> int:
    """
    Walk all checkpoint dirs under runs_dir and migrate any legacy-named .ckpt files.
    Returns count of files renamed.
    """
    count = 0
    if not runs_dir.exists():
        return count
    for ckpt in runs_dir.rglob("*.ckpt"):
        if ckpt.name in ("last.ckpt", "interrupted.ckpt"):
            continue
        result = _migrate_checkpoint_name(ckpt)
        if result != ckpt:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Raw keyboard input (cross-platform)
# ---------------------------------------------------------------------------

if IS_WINDOWS:
    import msvcrt

    def _getch() -> str:
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ext = msvcrt.getwch()
            return {"H": "UP", "P": "DOWN", "M": "RIGHT", "K": "LEFT"}.get(ext, "")
        return ch

else:
    def _getch() -> str:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                rest = sys.stdin.read(2)
                seq = ch + rest
                return {"[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT"}.get(rest, seq)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# ASCII banner
# ---------------------------------------------------------------------------

BANNER = r"""
    ___    ____  ____  __    __    ____     __  _______  ____
   /   |  / __ \/ __ \/ /   / /   / __ \   /  |/  / __ \/ __ \
  / /| | / /_/ / / / / /   / /   / / / /  / /|_/ / / / / / / /
 / ___ |/ ____/ /_/ / /___/ /___/ /_/ /  / /  / / /_/ / /_/ /
/_/  |_/_/    \____/_____/_____/\____/  /_/  /_/\____/_____/
"""

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _banner_panel() -> Panel:
    t = Text(BANNER, style="bold cyan", justify="center")
    return Panel(t, border_style="dim cyan", padding=(0, 2))


# Visible rows in the list viewport (keeps panel a fixed size)
_VIEWPORT_SIZE = 18


def _menu(
    title: str,
    items: list[str],
    selected: int = 0,
    hint: str = "",
    subtitle: str = "",
    viewport_top: int = 0,
) -> Panel:
    visible = items[viewport_top : viewport_top + _VIEWPORT_SIZE]
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True)
    for i, item in enumerate(visible):
        abs_i = viewport_top + i
        if abs_i == selected:
            row = Text(f"\u25b6  {item}", style="bold bright_white on grey23")
        else:
            row = Text(f"   {item}", style="dim white")
        table.add_row(row)

    # Scroll indicator
    n = len(items)
    if n > _VIEWPORT_SIZE:
        scroll_info = f"  {viewport_top+1}-{min(viewport_top+_VIEWPORT_SIZE, n)} of {n}"
    else:
        scroll_info = ""

    sub = Text(subtitle, style="dim") if subtitle else Text("")
    body = Table.grid()
    body.add_column()
    if subtitle:
        body.add_row(sub)
        body.add_row(Text(""))
    body.add_row(table)
    footer_parts = []
    if hint:
        footer_parts.append(hint)
    if scroll_info:
        footer_parts.append(scroll_info)
    if footer_parts:
        body.add_row(Text(""))
        body.add_row(Text("  ".join(footer_parts), style="dim italic"))

    return Panel(
        Align.left(body),
        title=f"[bold cyan]{title}[/]",
        border_style="cyan",
        padding=(1, 3),
    )


def _navigate(items: list[str], title: str, hint: str = "", subtitle: str = "", start: int = 0) -> int | None:
    """Show a viewport-scrolling menu and return selected index, or None on Escape/q."""
    n = len(items)
    if n == 0:
        return None
    sel = max(0, min(start, n - 1))
    vt  = max(0, sel - _VIEWPORT_SIZE // 2)  # centre viewport on start item

    def _clamp_vt(s, v):
        v = max(0, min(v, max(0, n - _VIEWPORT_SIZE)))
        # keep sel inside viewport
        if s < v:
            v = s
        elif s >= v + _VIEWPORT_SIZE:
            v = s - _VIEWPORT_SIZE + 1
        return v

    vt = _clamp_vt(sel, vt)

    with Live(_menu(title, items, sel, hint=hint, subtitle=subtitle, viewport_top=vt),
              console=console, auto_refresh=False, screen=False) as live:
        while True:
            ch = _getch()
            if ch in ("UP", "k"):
                sel = (sel - 1) % n
                vt  = _clamp_vt(sel, vt)
            elif ch in ("DOWN", "j"):
                sel = (sel + 1) % n
                vt  = _clamp_vt(sel, vt)
            elif ch in ("\r", "\n", " "):
                return sel
            elif ch in ("\x1b", "q", "Q"):
                return None
            elif ch == "\x03":  # Ctrl+C
                raise KeyboardInterrupt
            live.update(_menu(title, items, sel, hint=hint, subtitle=subtitle, viewport_top=vt), refresh=True)
    return None


def _pick(title: str, items: list[str], hint: str = "", subtitle: str = "", start: int = 0) -> int | None:
    """Wrapper that clears screen before showing menu."""
    console.clear()
    console.print(_banner_panel())
    return _navigate(items, title, hint=hint, subtitle=subtitle, start=start)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _list_configs() -> list[Path]:
    if not CONFIGS_DIR.exists():
        return []
    configs = list(CONFIGS_DIR.glob("*.yaml"))
    if os.environ.get("UNIVERSR_DEV") and DEV_CONFIGS_DIR.exists():
        configs += [p for p in DEV_CONFIGS_DIR.glob("*.yaml") if p.name != "AGENTS.md"]
    return sorted(configs, key=lambda p: p.stem)


def _ckpt_score(stem: str) -> tuple:
    """
    Composite score tuple for checkpoint ranking.
    Primary: visqol (higher better).
    Tiebreakers in order: sisdr, hfnr (lower = less noise).
    Returns a tuple suitable for max() comparison; missing metrics use worst-case values.
    """
    import re as _re
    stem = _re.sub(r"^\[\d+\]-", "", stem)
    def _get(pattern, default):
        m = _re.search(pattern, stem)
        try:
            return float(m.group(1)) if m else default
        except Exception:
            return default

    visqol = _get(r"visqol=(-?[\d.]+)", -999.0)
    sisdr  = _get(r"sisdr=(-?[\d.]+)",   -999.0)
    hfnr    = _get(r"hfnr=(-?[\d.]+)",     999.0)   # lower hfnr is better → negate

    # Existing checkpoints on disk have negated sisdr (legacy loss value); normalise
    # so that higher magnitude always wins regardless of which convention was used.
    if sisdr < 0:
        sisdr = -sisdr

    if visqol < 0:
        return None  # no visqol = unscored
    return (visqol, sisdr, -hfnr)


def _dedup_ckpts(ckpts):
    """
    Given an iterable of Path objects, return a deduplicated list where only
    the best-scoring checkpoint per step number is kept.  Duplicate step files
    arise from partial migrations that left old and new names both present.
    """
    import re as _re
    by_step = {}
    for ckpt in ckpts:
        m = _re.search(r"step=(\d+)", ckpt.stem)
        step = int(m.group(1)) if m else None
        score = _ckpt_score(ckpt.stem)
        key = (ckpt.parent, step)
        if key not in by_step or (score is not None and (by_step[key][1] is None or score > by_step[key][1])):
            by_step[key] = (ckpt, score)
    return [v[0] for v in by_step.values()]


def _config_summary(cfg_path: Path) -> str:
    """Return a one-line summary of training state for this config."""
    try:
        import yaml  # type: ignore
        import re as _re
        cfg = yaml.safe_load(cfg_path.read_text())
        name = cfg.get("exp", {}).get("name") or cfg_path.stem
        runs_path = RUNS_DIR / name
        if not runs_path.exists():
            return "no runs yet"
        best_score = None
        best_stem  = None
        best_step  = None
        for run_dir in sorted(runs_path.iterdir()):
            ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.exists():
                continue
            for ckpt in _dedup_ckpts(ckpt_dir.glob("*.ckpt")):
                score = _ckpt_score(ckpt.stem)
                if score is None:
                    continue
                s = _re.search(r"step=(\d+)", ckpt.stem)
                step = int(s.group(1)) if s else 0
                if best_score is None or score > best_score:
                    best_score = score
                    best_stem  = ckpt.stem
                    best_step  = step
        if best_score is not None:
            visqol, sisdr, neg_hfnr = best_score
            return (f"best visqol={visqol:.3f}  sisdr={sisdr:.2f}"
                    f"  hfnr={-neg_hfnr:.3f}  step={best_step}")
        return "checkpoint found (no metrics in name)"
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _find_best_checkpoint(cfg_path: Path) -> Path | None:
    """Find best checkpoint using composite score: visqol > sisdr > hfnr."""
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        name = cfg.get("exp", {}).get("name") or cfg_path.stem
        runs_path = RUNS_DIR / name
        if not runs_path.exists():
            return None
        best_score = None
        best_ckpt  = None
        for run_dir in sorted(runs_path.iterdir()):
            ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.exists():
                continue
            for ckpt in _dedup_ckpts(ckpt_dir.glob("*.ckpt")):
                score = _ckpt_score(ckpt.stem)
                if score is None:
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_ckpt  = ckpt
        return best_ckpt
    except Exception:
        return None


def _find_latest_checkpoint(cfg_path: Path) -> Path | None:
    """Find the most recently saved checkpoint for a config (by mtime)."""
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        name = cfg.get("exp", {}).get("name") or cfg_path.stem
        runs_path = RUNS_DIR / name
        if not runs_path.exists():
            return None
        all_ckpts = []
        for run_dir in sorted(runs_path.iterdir()):
            ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.exists():
                continue
            all_ckpts.extend(ckpt_dir.glob("*.ckpt"))
        if not all_ckpts:
            return None
        return max(all_ckpts, key=lambda p: p.stat().st_mtime)
    except Exception:
        return None


def _find_model_for_config(cfg_path: Path) -> Path | None:
    """Look for a .ckpt or .pth in /models matching config name."""
    if not MODELS_DIR.exists():
        return None
    stem = cfg_path.stem
    for ext in (".ckpt", ".pth"):
        candidate = MODELS_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Subprocess runner (live output, returns to menu on Ctrl+C)
# ---------------------------------------------------------------------------

def _run_mid_training_inference(state: dict, cfg_path: Path, pause_file: Path) -> None:
    """Mini inference flow triggered by Ctrl+I during training."""
    import re as _re
    console.clear()
    console.print(_banner_panel())
    console.print(Panel(
        "[bold yellow]Mid-training Inference[/]\n[dim]Training is paused. Pick a file to process.[/]",
        border_style="yellow",
        padding=(0, 2),
    ))

    cfg_stem = cfg_path.stem

    # Pick input
    input_path = _pick_input_file(state, cfg_stem)
    if not input_path or input_path == _PROCESS_ALL_SENTINEL:
        # If cancelled or process-all, just resume
        pause_file.unlink(missing_ok=True)
        return

    # Pick output
    output_path = _pick_output_path(state, cfg_stem, input_path)
    if not output_path:
        pause_file.unlink(missing_ok=True)
        return

    # Find latest checkpoint for this config
    latest_ckpt = _find_latest_checkpoint(cfg_path)
    if latest_ckpt is None:
        console.print("[red]No checkpoint found yet -- cannot run inference.[/]")
        console.input("[dim]Press Enter to resume training[/]")
        pause_file.unlink(missing_ok=True)
        return

    # Build and run inference cmd
    import re as _re2
    stem = _re2.sub(r"^\[\d+\]-", "", latest_ckpt.stem)
    m = _re2.search(r"visqol=(-?[\d.]+)", stem)
    visqol_str = f"visqol={float(m.group(1)):.3f}" if (m and float(m.group(1)) >= 0) else ""

    try:
        cfg_data = __import__("yaml").safe_load(cfg_path.read_text())
        feature_dim = cfg_data.get("model", {}).get("feature_dim", 256)
    except Exception:
        feature_dim = 384

    cmd = [
        _python_bin(), str(ROOT / "core" / "inference.py"),
        "--in_wav",    input_path,
        "--out_wav",   output_path,
        "--weights",   str(latest_ckpt),
        "--conf_dir",  str(cfg_path),
        "--feature_dim", str(feature_dim),
    ]

    console.print(f"\n[cyan]Checkpoint:[/] {latest_ckpt.name}  {visqol_str}")
    console.print(f"[cyan]Input:[/]      {Path(input_path).name}")
    console.print(f"[cyan]Output:[/]     {output_path}\n")

    _run_subprocess(cmd, "Mid-training Inference", pause_dir=None)

    console.print("\n[green]Inference done. Resuming training...[/]")
    import time as _t; _t.sleep(1)
    pause_file.unlink(missing_ok=True)


def _run_subprocess(cmd: list[str], label: str, pause_dir: Path | None = None,
                    state: dict | None = None, cfg_path: Path | None = None) -> bool:
    """Run a subprocess, stream output. Returns True if completed, False if Ctrl+C.

    When pause_dir is set, watches for Ctrl+I (ASCII 0x09) to trigger mid-training
    inference. Writes pause_dir/PAUSED to signal train.py to suspend itself.
    """
    console.clear()
    console.print(_banner_panel())
    hint = "[dim]Press Ctrl+C to stop  |  Ctrl+I to pause and run inference[/]" if pause_dir else "[dim]Press Ctrl+C to stop and return to menu[/]"
    console.print(Panel(
        f"[bold cyan]{label}[/]\n{hint}",
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print(Rule(style="dim cyan"))

    proc = None
    interrupted = False
    pause_file = (pause_dir / "PAUSED") if pause_dir else None
    _pausing = threading.Event()

    def _watch_for_ctrl_i():
        """Background thread: read raw stdin bytes, trigger pause on Ctrl+I (0x09)."""
        if not pause_dir or not sys.stdin.isatty():
            return
        try:
            if IS_WINDOWS:
                import msvcrt
                while proc and proc.poll() is None:
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch == "\t":  # Tab = Ctrl+I on Windows
                            _pausing.set()
                            break
                    import time as _t; _t.sleep(0.05)
            else:
                import tty as _tty, termios as _termios
                fd = sys.stdin.fileno()
                old = _termios.tcgetattr(fd)
                try:
                    _tty.setraw(fd)
                    while proc and proc.poll() is None:
                        import select
                        r, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if r:
                            ch = sys.stdin.read(1)
                            if ch == "\t":  # Ctrl+I
                                _pausing.set()
                                break
                finally:
                    _termios.tcsetattr(fd, _termios.TCSADRAIN, old)
        except Exception:
            pass

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Insert -u after the interpreter if the command starts with a Python binary,
        # so the child process flushes stdout per line even when writing to a pipe.
        _cmd = list(cmd)
        if _cmd and _cmd[0].endswith(("python", "python3", "python.exe")):
            _cmd.insert(1, "-u")

        proc = subprocess.Popen(
            _cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(ROOT),
        )

        def _stream():
            assert proc and proc.stdout
            for line in proc.stdout:
                if not _pausing.is_set():
                    sys.stdout.write(line)
                    sys.stdout.flush()

        t = threading.Thread(target=_stream, daemon=True)
        t.start()

        if pause_dir:
            ki = threading.Thread(target=_watch_for_ctrl_i, daemon=True)
            ki.start()

        while proc.poll() is None:
            if _pausing.is_set() and pause_dir and pause_file and cfg_path and state is not None:
                pause_file.touch()
                _run_mid_training_inference(state, cfg_path, pause_file)
                _pausing.clear()
                # Reprint header after returning
                console.print(Rule(style="dim cyan"))
            import time as _t; _t.sleep(0.05)

        t.join(timeout=5)

    except KeyboardInterrupt:
        interrupted = True
        if proc and proc.poll() is None:
            console.print("\n[yellow]Ctrl+C caught -- sending stop signal...[/]")
            if IS_WINDOWS:
                proc.send_signal(signal.CTRL_C_EVENT)  # type: ignore[attr-defined]
            else:
                proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        console.print("[dim]Returning to menu...[/]")
    finally:
        if pause_file and pause_file.exists():
            pause_file.unlink(missing_ok=True)

    console.print(Rule(style="dim cyan"))
    return not interrupted


def _run_with_live_output(cmd: list[str], label: str,
                          pause_dir: Path | None = None,
                          state: dict | None = None,
                          cfg_path: Path | None = None) -> None:
    """Run a subprocess and wait for Enter before returning to menu."""
    _run_subprocess(cmd, label, pause_dir=pause_dir, state=state, cfg_path=cfg_path)
    console.input("[dim]Press Enter to return to menu[/]")


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

def _python_bin() -> str:
    venv = ROOT / ".venv"
    if IS_WINDOWS:
        return str(venv / "Scripts" / "python.exe")
    return str(venv / "bin" / "python")


# -- Train -------------------------------------------------------------------

def screen_train(state: dict) -> None:
    configs = _list_configs()
    if not configs:
        console.clear()
        console.print("[red]No configs found in configs/[/]")
        console.input("Press Enter to return.")
        return

    last_cfg = state.get("train", {}).get("last_config", "")
    start = next((i for i, c in enumerate(configs) if c.name == last_cfg), 0)

    items = [c.stem for c in configs]
    summaries = [_config_summary(c) for c in configs]

    while True:
        # build display items with summary
        display = [f"{items[i]}  [dim]{summaries[i]}[/dim]" for i in range(len(items))]
        # Rich doesn't render markup in Text directly in menu, so strip for plain display
        plain = [f"{items[i]}   {summaries[i]}" for i in range(len(items))]
        idx = _pick("Train -- select config", plain, hint="Enter=start  Esc=back", start=start)
        if idx is None:
            return

        cfg_path = configs[idx]
        state.setdefault("train", {})["last_config"] = cfg_path.name
        _save_state(state)

        cmd = [_python_bin(), str(ROOT / "core" / "train.py"), "--conf_dir", str(cfg_path)]
        # Derive run dir for pause file -- matches train.py's run isolation logic.
        # We use the base dir; train.py will pick the right timestamped subfolder.
        # The pause file goes in the base exp dir so train.py can always find it.
        try:
            import yaml as _yaml
            _cfg_data = _yaml.safe_load(cfg_path.read_text())
            _exp_dir  = _cfg_data.get("exp", {}).get("dir", "./runs")
            _exp_name = _cfg_data.get("exp", {}).get("name") or cfg_path.stem
            _pause_dir = ROOT / _exp_dir / _exp_name
        except Exception:
            _pause_dir = None
        _run_with_live_output(cmd, f"Training: {cfg_path.stem}",
                              pause_dir=_pause_dir, state=state, cfg_path=cfg_path)
        start = idx
        return


# -- Inference ---------------------------------------------------------------

_PROCESS_ALL_SENTINEL = "__PROCESS_ALL__"


def _pick_input_file(state: dict, cfg_stem: str) -> str | None:
    """Pick an input audio file from /input, process all, or enter custom path.

    Returns a file path string, _PROCESS_ALL_SENTINEL, or None to cancel.
    """
    last = state.get("inference", {}).get(cfg_stem, {}).get("last_input", "")

    INPUT_DIR.mkdir(exist_ok=True)
    files = sorted(f for f in INPUT_DIR.iterdir() if f.suffix.lower() in AUDIO_EXTS)
    file_names = [f.name for f in files]

    process_all_label = (
        f"[ Process all {len(files)} file(s) in /input -> /output ]"
        if files else "[ Process all in /input (folder empty) ]"
    )
    # Process-all and custom path go at the TOP so they're always reachable
    # without scrolling, regardless of how many files are in /input.
    PROCESS_ALL_IDX = 0
    CUSTOM_IDX      = 1
    items = [process_all_label, "[ Enter custom path ]"] + file_names

    # Offset saved-last-input index by 2 to account for the two header items
    last_name = Path(last).name
    file_start = next((i for i, n in enumerate(file_names) if n == last_name), None)
    start = (file_start + 2) if file_start is not None else 2

    idx = _pick(
        "Inference -- select input",
        items,
        hint="Enter=select  Esc=back",
        start=start,
    )
    if idx is None:
        return None

    if idx == PROCESS_ALL_IDX:
        return _PROCESS_ALL_SENTINEL

    if idx == CUSTOM_IDX:
        console.clear()
        console.print(_banner_panel())
        path = console.input("[cyan]Enter path to input file:[/] ").strip().strip('"')
        return path if path else None

    return str(files[idx - 2])


def _pick_output_path(state: dict, cfg_stem: str, input_path: str) -> str | None:
    """Pick output path -- default to /output/<input_stem>_restored.wav or last used."""
    last = state.get("inference", {}).get(cfg_stem, {}).get("last_output", "")
    input_stem = Path(input_path).stem
    default = str(OUTPUT_DIR / f"{input_stem}_restored.wav")
    suggested = last if last else default

    items = [
        f"Default: {suggested}",
        "[ Enter custom path ]",
    ]
    idx = _pick("Inference -- output path", items, hint="Enter=select  Esc=back")
    if idx is None:
        return None
    if idx == 0:
        OUTPUT_DIR.mkdir(exist_ok=True)
        return suggested
    console.clear()
    console.print(_banner_panel())
    path = console.input("[cyan]Enter output path:[/] ").strip().strip('"')
    return path if path else None


def _pick_ensemble(state: dict, cfg_stem: str) -> tuple:
    """Pick ensemble / spectral merge options.

    Returns (extra_flags: list[str], label: str) where extra_flags are the
    CLI args to append to the inference command, and label is a short
    human-readable description of what was chosen.
    """
    import json as _json

    options = [
        "No ensemble  (model output only)",
        "Low-end preserve  (max_fft below 700 Hz)",
        "Low-end preserve + transition blend  (max_fft <700 Hz, avg 15-22 kHz, weight 0.6)",
        "Custom  (enter JSON band spec)",
    ]

    last_idx = state.get("inference", {}).get(cfg_stem, {}).get("last_ensemble_idx", 0)
    idx = _pick("Inference -- ensemble / spectral merge", options,
                hint="Enter=select  Esc=back (skip ensemble)", start=last_idx)
    if idx is None:
        return [], "no ensemble"

    state.setdefault("inference", {}).setdefault(cfg_stem, {})["last_ensemble_idx"] = idx

    if idx == 0:
        return [], "no ensemble"
    if idx == 1:
        return ["--low_end_preserve"], "low-end preserve"
    if idx == 2:
        bands = [
            {"lo": 0,     "hi": 700,   "mode": "max_fft", "weight": 1.0},
            {"lo": 15000, "hi": 22050, "mode": "avg",     "weight": 0.6},
        ]
        return ["--ensemble", _json.dumps(bands)], "low-end + transition blend"

    # Custom JSON input
    console.clear()
    console.print(_banner_panel())
    console.print("[dim]Band spec format: [{\"lo\":Hz,\"hi\":Hz,\"mode\":\"max_fft|min_fft|avg|original|enhanced\",\"weight\":0-1}][/]")
    raw = console.input("[cyan]Enter JSON band spec:[/] ").strip()
    if not raw:
        return [], "no ensemble"
    try:
        bands = _json.loads(raw)
        return ["--ensemble", _json.dumps(bands)], "custom ensemble"
    except Exception as e:
        console.print(f"[red]Invalid JSON: {e}[/]")
        console.input("Press Enter.")
        return [], "no ensemble"


def screen_inference(state: dict) -> None:
    configs = _list_configs()
    if not configs:
        console.clear()
        console.print("[red]No configs found in configs/[/]")
        console.input("Press Enter to return.")
        return

    last_cfg = state.get("inference", {}).get("last_config", "")
    start = next((i for i, c in enumerate(configs) if c.name == last_cfg), 0)

    items = [c.stem for c in configs]
    idx = _pick("Inference -- select config", items, hint="Enter=select  Esc=back", start=start)
    if idx is None:
        return

    cfg_path = configs[idx]
    cfg_stem = cfg_path.stem
    state.setdefault("inference", {})["last_config"] = cfg_path.name
    _save_state(state)

    # Find checkpoint / model
    best_ckpt = _find_best_checkpoint(cfg_path)
    model_file = _find_model_for_config(cfg_path)

    # Build model options
    model_options = []
    model_paths = []

    import re as _re2
    latest_ckpt = _find_latest_checkpoint(cfg_path)
    if latest_ckpt:
        stem = _re2.sub(r"^\[\d+\]-", "", latest_ckpt.stem)
        m = _re2.search(r"visqol=(-?[\d.]+)", stem)
        visqol_str = f"visqol={float(m.group(1)):.3f}" if (m and float(m.group(1)) >= 0) else ""
        model_options.append(f"Latest checkpoint  {visqol_str}  ({latest_ckpt.name})")
        model_paths.append(str(latest_ckpt))
    if best_ckpt and (not latest_ckpt or best_ckpt != latest_ckpt):
        stem = _re2.sub(r"^\[\d+\]-", "", best_ckpt.stem)
        m = _re2.search(r"visqol=(-?[\d.]+)", stem)
        visqol_str = f"visqol={float(m.group(1)):.3f}" if (m and float(m.group(1)) >= 0) else ""
        model_options.append(f"Best checkpoint  {visqol_str}  ({best_ckpt.name})")
        model_paths.append(str(best_ckpt))
    if model_file:
        model_options.append(f"Model file  ({model_file.name})")
        model_paths.append(str(model_file))
    model_options.append("[ Enter custom weights path ]")

    last_weights = state.get("inference", {}).get(cfg_stem, {}).get("last_weights", "")
    w_start = next((i for i, p in enumerate(model_paths) if p == last_weights), 0)

    widx = _pick(
        "Inference -- select model",
        model_options,
        hint="Enter=select  Esc=back",
        start=w_start,
    )
    if widx is None:
        return

    if widx == len(model_options) - 1:
        console.clear()
        console.print(_banner_panel())
        weights = console.input("[cyan]Enter path to weights:[/] ").strip().strip('"')
        if not weights:
            return
    else:
        weights = model_paths[widx]

    state.setdefault("inference", {}).setdefault(cfg_stem, {})["last_weights"] = weights
    _save_state(state)

    # Pick input
    input_path = _pick_input_file(state, cfg_stem)
    if not input_path:
        return

    # Pick ensemble options (applies to both single and batch)
    ensemble_flags, ensemble_label = _pick_ensemble(state, cfg_stem)
    _save_state(state)

    # --- Batch mode: process all files in /input ---
    if input_path == _PROCESS_ALL_SENTINEL:
        INPUT_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)
        batch_files = sorted(
            f for f in INPUT_DIR.iterdir() if f.suffix.lower() in AUDIO_EXTS
        )
        if not batch_files:
            console.clear()
            console.print(_banner_panel())
            console.print("[yellow]No audio files found in /input.[/]")
            console.input("Press Enter.")
            return
        for i, in_file in enumerate(batch_files):
            out_file = OUTPUT_DIR / f"{in_file.stem}_restored.wav"
            cmd = [
                _python_bin(), str(ROOT / "core" / "inference.py"),
                "--in_wav", str(in_file),
                "--out_wav", str(out_file),
                "--conf_dir", str(cfg_path),
                "--weights", weights,
            ] + ensemble_flags
            completed = _run_subprocess(
                cmd,
                f"Inference [{i+1}/{len(batch_files)}]: {in_file.name}  [{ensemble_label}]",
            )
            if not completed:
                # User hit Ctrl+C mid-batch -- stop processing remaining files
                console.print(f"[yellow]Batch stopped after {i}/{len(batch_files)} files.[/]")
                console.input("[dim]Press Enter to return to menu[/]")
                return
        console.print(f"[green]Batch complete: {len(batch_files)} file(s) processed.[/]")
        console.input("[dim]Press Enter to return to menu[/]")
        return

    # --- Single file mode ---
    state["inference"][cfg_stem]["last_input"] = input_path
    _save_state(state)

    # Pick output
    output_path = _pick_output_path(state, cfg_stem, input_path)
    if not output_path:
        return

    state["inference"][cfg_stem]["last_output"] = output_path
    _save_state(state)

    # Build command
    cmd = [
        _python_bin(), str(ROOT / "core" / "inference.py"),
        "--in_wav", input_path,
        "--out_wav", output_path,
        "--conf_dir", str(cfg_path),
        "--weights", weights,
    ] + ensemble_flags
    _run_with_live_output(cmd, f"Inference: {cfg_path.stem}  [{ensemble_label}]")


# -- Edit Config -------------------------------------------------------------

def _flatten_yaml(data: dict, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten nested yaml into dotted key paths, skipping _target_ and internal keys."""
    SKIP = {"_target_", "sr", "win", "layer"}
    result = []
    for k, v in data.items():
        if k in SKIP:
            continue
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.extend(_flatten_yaml(v, full_key))
        else:
            result.append((full_key, v))
    return result


def _set_nested(data: dict, dotted_key: str, value: object) -> None:
    keys = dotted_key.split(".")
    d = data
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def _parse_value(s: str, original) -> object:
    """Parse string back to the same type as original."""
    if isinstance(original, bool):
        return s.lower() in ("true", "yes", "1")
    if isinstance(original, int):
        return int(s)
    if isinstance(original, float):
        return float(s)
    if isinstance(original, list):
        import ast
        return ast.literal_eval(s)
    # string or None
    if s.lower() in ("null", "none", "false"):
        return False if isinstance(original, bool) else (None if s.lower() in ("null", "none") else s)
    return s


def screen_edit_config(state: dict) -> None:
    try:
        import yaml
    except ImportError:
        console.print("[red]PyYAML not available[/]")
        console.input("Press Enter.")
        return

    configs = _list_configs()
    if not configs:
        console.clear()
        console.print("[red]No configs found[/]")
        console.input("Press Enter.")
        return

    last_cfg = state.get("edit", {}).get("last_config", "")
    start = next((i for i, c in enumerate(configs) if c.name == last_cfg), 0)
    items = [c.stem for c in configs]

    idx = _pick("Edit Config -- select config", items, hint="Enter=select  Esc=back", start=start)
    if idx is None:
        return

    cfg_path = configs[idx]
    state.setdefault("edit", {})["last_config"] = cfg_path.name
    _save_state(state)

    data = yaml.safe_load(cfg_path.read_text())
    pairs = _flatten_yaml(data)
    keys = [f"{k}  =  {v}" for k, v in pairs]

    sel = 0
    while True:
        kidx = _pick(
            f"Edit: {cfg_path.stem}",
            keys,
            hint="Enter=edit value  Esc=back (auto-saves)",
            start=sel,
        )
        if kidx is None:
            # save on exit
            with open(cfg_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            return

        sel = kidx
        dotted_key, original_value = pairs[kidx]

        console.clear()
        console.print(_banner_panel())
        console.print(Panel(
            f"[bold]{dotted_key}[/]\nCurrent value: [cyan]{original_value}[/]",
            border_style="cyan",
        ))
        new_str = console.input("[cyan]New value[/] (Enter to keep): ").strip()
        if new_str:
            try:
                new_val = _parse_value(new_str, original_value)
                _set_nested(data, dotted_key, new_val)
                pairs[kidx] = (dotted_key, new_val)
                keys[kidx] = f"{dotted_key}  =  {new_val}"
            except Exception as e:
                console.print(f"[red]Invalid value: {e}[/]")
                console.input("Press Enter.")


# -- Utilities ---------------------------------------------------------------

def _get_chunk_dirs() -> list[tuple[str, Path]]:
    chunks_root = ROOT / "chunks"
    if not chunks_root.exists():
        return []
    return [(d.name, d) for d in sorted(chunks_root.iterdir()) if d.is_dir()]


def _get_run_dirs() -> list[tuple[str, Path]]:
    if not RUNS_DIR.exists():
        return []
    result = []
    for cfg_dir in sorted(RUNS_DIR.iterdir()):
        if cfg_dir.is_dir():
            result.append((cfg_dir.name, cfg_dir))
    return result


def screen_utilities(state: dict) -> None:
    while True:
        items = [
            "Clean chunks folder",
            "Clean old checkpoints (keep best 5)",
            "View training runs",
            "Degrade audio",
            "Align audio",
            "Update Apollo",
            "Back",
        ]
        idx = _pick("Utilities", items, hint="Enter=select  Esc=back")
        if idx is None or idx == len(items) - 1:
            return

        if idx == 0:
            _util_clean_chunks()
        elif idx == 1:
            _util_clean_checkpoints()
        elif idx == 2:
            _util_view_runs()
        elif idx == 3:
            from degrade_audio import screen_degrade_audio
            screen_degrade_audio(state, console, _pick, _run_with_live_output, ROOT)
        elif idx == 4:
            from align_audio import screen_align_audio
            screen_align_audio(state, console, _pick, _run_with_live_output, ROOT)
        elif idx == 5:
            _util_update()


def _util_update() -> None:
    """Pull the latest code via git, then relaunch the TUI in place."""
    console.clear()
    console.print(_banner_panel())

    if not (ROOT / ".git").exists():
        console.print("[red]Not a git checkout -- cannot update.[/]")
        console.input("Press Enter to return.")
        return

    import shutil
    git_bin = shutil.which("git")
    if not git_bin:
        console.print("[red]git not found on PATH -- cannot update.[/]")
        console.input("Press Enter to return.")
        return

    console.print("[cyan]Checking for updates...[/]\n")
    result = subprocess.run(
        [git_bin, "pull", "--ff-only"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    console.print(result.stdout)
    if result.returncode != 0:
        console.print(result.stderr)
        console.print("[red]Update failed -- local copy unchanged.[/]")
        console.input("Press Enter to return.")
        return

    if "Already up to date" in result.stdout:
        console.print("[green]Already up to date.[/]")
        console.input("Press Enter to return.")
        return

    console.print("[green]Updated. Relaunching...[/]")
    console.file.flush()
    python = sys.executable
    os.execv(python, [python, str(ROOT / "utils" / "tui.py")])


def _util_clean_chunks() -> None:
    dirs = _get_chunk_dirs()
    if not dirs:
        console.clear()
        console.print("[dim]No chunk folders found.[/]")
        console.input("Press Enter.")
        return

    items = [f"{name}  ({_dir_size(path)})" for name, path in dirs] + ["Back"]
    idx = _pick("Clean Chunks -- select folder to delete", items, hint="Enter=delete  Esc=back")
    if idx is None or idx == len(items) - 1:
        return

    name, path = dirs[idx]
    console.clear()
    console.print(_banner_panel())
    confirm = console.input(f"[red]Delete chunks/{name}? (yes/no):[/] ").strip().lower()
    if confirm == "yes":
        import shutil
        shutil.rmtree(path)
        console.print(f"[green]Deleted chunks/{name}[/]")
    else:
        console.print("[dim]Cancelled.[/]")
    console.input("Press Enter.")


def _util_clean_checkpoints() -> None:
    run_dirs = _get_run_dirs()
    if not run_dirs:
        console.clear()
        console.print("[dim]No run folders found.[/]")
        console.input("Press Enter.")
        return

    console.clear()
    console.print(_banner_panel())
    removed = 0
    for cfg_name, cfg_dir in run_dirs:
        for run_dir in sorted(cfg_dir.iterdir()):
            ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.exists():
                continue
            all_ckpts = list(ckpt_dir.glob("*.ckpt"))
            # Score each checkpoint; unscored ones go to the bottom
            scored = []
            for c in all_ckpts:
                score = _ckpt_score(c.stem)
                scored.append((score or (-999.0, -999.0, -999.0, -999.0), c))
            # Sort best-first, keep top 5
            scored.sort(key=lambda x: x[0], reverse=True)
            to_delete = [c for _, c in scored[5:]]
            for c in to_delete:
                c.unlink()
                removed += 1
    console.print(f"[green]Removed {removed} checkpoint(s).[/]")
    console.input("Press Enter.")


def _util_view_runs() -> None:
    console.clear()
    console.print(_banner_panel())

    table = Table(title="Training Runs", border_style="dim cyan", show_lines=True)
    table.add_column("Config", style="cyan")
    table.add_column("Run", style="dim")
    table.add_column("Best sisdr", style="green")
    table.add_column("Step")
    table.add_column("Checkpoints")

    if RUNS_DIR.exists():
        for cfg_dir in sorted(RUNS_DIR.iterdir()):
            if not cfg_dir.is_dir():
                continue
            for run_dir in sorted(cfg_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                ckpt_dir = run_dir / "checkpoints"
                if not ckpt_dir.exists():
                    continue
                ckpts = list(ckpt_dir.glob("*.ckpt"))
                best_loss = None
                best_step = None
                for c in ckpts:
                    if "sisdr=" in c.stem:
                        try:
                            loss = float(c.stem.split("sisdr=")[1])
                            step = int(c.stem.split("step=")[1].split("-")[0])
                            if best_loss is None or loss < best_loss:
                                best_loss = loss
                                best_step = step
                        except Exception:
                            pass
                table.add_row(
                    cfg_dir.name,
                    run_dir.name,
                    f"{best_loss:.4f}" if best_loss else "--",
                    str(best_step) if best_step else "--",
                    str(len(ckpts)),
                )

    console.print(table)
    console.input("\nPress Enter to return.")


def _dir_size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    if total > 1_073_741_824:
        return f"{total/1_073_741_824:.1f} GB"
    if total > 1_048_576:
        return f"{total/1_048_576:.0f} MB"
    return f"{total/1024:.0f} KB"


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def screen_evaluate(state: dict) -> None:
    from evaluate import screen_evaluate as _screen_evaluate
    _screen_evaluate(state, console, _pick, _run_with_live_output, ROOT)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    migrate_checkpoints(RUNS_DIR)
    state = _load_state()

    MAIN_ITEMS = [
        "Train",
        "Inference",
        "Evaluate",
        "Edit Config",
        "Utilities",
        "Exit",
    ]

    sel = 0
    while True:
        idx = _pick("Main Menu", MAIN_ITEMS, hint="^v navigate  Enter select  q quit", start=sel)
        if idx is None or idx == len(MAIN_ITEMS) - 1:
            console.clear()
            break
        sel = idx
        if idx == 0:
            screen_train(state)
        elif idx == 1:
            screen_inference(state)
        elif idx == 2:
            screen_evaluate(state)
        elif idx == 3:
            screen_edit_config(state)
        elif idx == 4:
            screen_utilities(state)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.clear()
        sys.exit(0)
