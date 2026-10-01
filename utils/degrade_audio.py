"""
degrade_audio.py -- Synthetic audio degradation pipeline for training data gen.

Replaces degrade_audio.bat. The codec/filter chain is defined in a JSON config
under utils/degrade/ instead of being hardcoded, so different degradation
recipes can be authored without editing code.

Launched from tui.py as a TUI screen, or standalone:
    python utils/degrade_audio.py --config utils/degrade/default.json --input "in.flac" --output "C:\\out"
    python utils/degrade_audio.py --config utils/degrade/default.json --input "C:\\in_folder" --output "C:\\out" --bulk

Chain step types:
    wma_encode   -- {bitrate}                          ffmpeg -c:a wmav2
    mp3_lame     -- {quality} or {bitrate}              ffmpeg -c:a libmp3lame
    mp3_fhg      -- {bitrate, enc_delay, codec_name}     acmenc (Fraunhofer IIS) --
                    requires "external_codecs.acmenc" set to a local acmenc.exe
                    path in the config; acmenc is a custom install, not shipped
    lowpass      -- {cutoff_hz, poles}                   ffmpeg -af lowpass
    highpass     -- {cutoff_hz, poles}                   ffmpeg -af highpass

Each step operates on the WAV output of the previous step. Steps that need a
compressed intermediate (mp3_lame, mp3_fhg, wma_encode) are followed
automatically by a decode-to-WAV pass before the next step -- the config only
lists the meaningful degradation passes, not the plumbing between them.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".aiff", ".aif"}

FHG_CODEC_NAME = "Fraunhofer IIS MPEG Layer-3 Codec (professional)"


class DegradeError(RuntimeError):
    pass


def _sanitize_basename(name: str) -> str:
    for ch in "()&":
        name = name.replace(ch, "")
    return name.replace(" ", "_")


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    cfg = json.loads(path.read_text())
    cfg.setdefault("tools", {})
    cfg["tools"].setdefault("ffmpeg", "ffmpeg")
    cfg.setdefault("external_codecs", {})
    if "chain" not in cfg or not cfg["chain"]:
        raise DegradeError(f"Config {path} has no 'chain' steps.")
    return cfg


def list_configs(configs_dir: Path) -> list[Path]:
    if not configs_dir.exists():
        return []
    return sorted(configs_dir.glob("*.json"))


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------

def _run(cmd: list[str], print_fn) -> None:
    print_fn(f"    $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print_fn(result.stdout)
        print_fn(result.stderr)
        raise DegradeError(f"Command failed (exit {result.returncode}): {cmd[0]}")


def _step_wma_encode(step: dict, src: Path, tmpdir: Path, n: int, tools: dict, print_fn) -> Path:
    bitrate = step.get("bitrate", "128k")
    out = tmpdir / f"gen{n}_wma.wma"
    _run([tools["ffmpeg"], "-y", "-i", str(src), "-vn", "-c:a", "wmav2", "-b:a", str(bitrate), str(out)], print_fn)
    return out


def _step_mp3_lame(step: dict, src: Path, tmpdir: Path, n: int, tools: dict, print_fn) -> Path:
    out = tmpdir / f"gen{n}_mp3.mp3"
    cmd = [tools["ffmpeg"], "-y", "-i", str(src), "-vn", "-c:a", "libmp3lame"]
    if "quality" in step:
        cmd += ["-q:a", str(step["quality"])]
    elif "bitrate" in step:
        cmd += ["-b:a", str(step["bitrate"])]
    else:
        raise DegradeError("mp3_lame step needs 'quality' or 'bitrate'.")
    cmd.append(str(out))
    _run(cmd, print_fn)
    return out


def _step_mp3_fhg(step: dict, src_wav: Path, tmpdir: Path, n: int, external_codecs: dict, print_fn, outdir_override: Path | None = None) -> Path:
    bitrate = step.get("bitrate", 192)
    enc_delay = step.get("enc_delay", 672)
    codec_name = step.get("codec_name", FHG_CODEC_NAME)
    out = (outdir_override / f"gen{n}_mp3.mp3") if outdir_override else (tmpdir / f"gen{n}_mp3.mp3")
    acmenc = external_codecs.get("acmenc")
    if not acmenc:
        raise DegradeError(
            "mp3_fhg step requires 'external_codecs.acmenc' in the config -- "
            "set it to the path to acmenc.exe on this machine. acmenc is a "
            "custom local install and is not shipped with this repo."
        )
    if not shutil.which(acmenc) and not Path(acmenc).exists():
        raise DegradeError(f"acmenc not found at '{acmenc}' -- check external_codecs.acmenc in the config.")
    cmd = [
        acmenc, "-c", codec_name,
        "--enc-delay", str(enc_delay),
        f"-b{bitrate}", str(src_wav), str(out),
    ]
    _run(cmd, print_fn)
    return out


def _step_filter(step: dict, kind: str, src: Path, tmpdir: Path, n: int, tools: dict, print_fn) -> Path:
    cutoff = step.get("cutoff_hz")
    if cutoff is None:
        raise DegradeError(f"{kind} step needs 'cutoff_hz'.")
    poles = step.get("poles", 2)
    out = tmpdir / f"gen{n}_{kind}.wav"
    af = f"{kind}=f={cutoff}:poles={poles}"
    _run([tools["ffmpeg"], "-y", "-i", str(src), "-vn", "-af", af, "-c:a", "pcm_s16le", str(out)], print_fn)
    return out


def _decode_to_wav(src: Path, tmpdir: Path, n: int, tools: dict, print_fn) -> Path:
    out = tmpdir / f"gen{n}_dec.wav"
    _run([tools["ffmpeg"], "-y", "-i", str(src), "-vn", "-c:a", "pcm_s16le", str(out)], print_fn)
    return out


_COMPRESSED_STEPS = {"wma_encode", "mp3_lame", "mp3_fhg"}
_FILTER_STEPS = {"lowpass", "highpass"}


def run_chain(config: dict, input_path: str | Path, output_dir: str | Path, print_fn=print) -> Path:
    """Run the configured codec/filter chain on a single file. Returns the output path."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise DegradeError(f"Input not found: {input_path}")

    tools = config["tools"]
    external_codecs = config["external_codecs"]
    chain = config["chain"]
    basename = _sanitize_basename(input_path.stem)

    output_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = output_dir / f"tmp_{basename}"
    tmpdir.mkdir(parents=True, exist_ok=True)

    print_fn(f"Degradation pipeline starting")
    print_fn(f"  Input:  {input_path}")
    print_fn(f"  Output: {output_dir}")
    print_fn(f"  Config: {config.get('name', '(unnamed)')}  ({len(chain)} steps)")
    print_fn("")

    try:
        current = input_path
        for i, step in enumerate(chain, start=1):
            step_type = step.get("type")
            print_fn(f"[{i}/{len(chain)}] {step_type} ...")

            is_last = (i == len(chain))

            if step_type == "wma_encode":
                current = _step_wma_encode(step, current, tmpdir, i, tools, print_fn)
            elif step_type == "mp3_lame":
                current = _step_mp3_lame(step, current, tmpdir, i, tools, print_fn)
            elif step_type == "mp3_fhg":
                # mp3_fhg needs WAV input; decode first if the previous step left compressed audio.
                if current.suffix.lower() != ".wav":
                    current = _decode_to_wav(current, tmpdir, i, tools, print_fn)
                dest_dir = output_dir if is_last else None
                current = _step_mp3_fhg(step, current, tmpdir, i, external_codecs, print_fn, outdir_override=dest_dir)
                if is_last:
                    final = output_dir / f"{basename}_degraded.mp3"
                    if current != final:
                        shutil.move(str(current), str(final))
                    current = final
            elif step_type in _FILTER_STEPS:
                if current.suffix.lower() != ".wav":
                    current = _decode_to_wav(current, tmpdir, i, tools, print_fn)
                current = _step_filter(step, step_type, current, tmpdir, i, tools, print_fn)
            else:
                raise DegradeError(f"Unknown step type: {step_type!r}")

            print_fn("    Done.")

        # If the chain didn't end on mp3_fhg (which writes straight to output_dir),
        # copy/convert whatever we ended up with into the final output file.
        last_type = chain[-1].get("type")
        if last_type != "mp3_fhg":
            ext = ".mp3" if last_type in ("wma_encode", "mp3_lame") else ".wav"
            final = output_dir / f"{basename}_degraded{ext}"
            shutil.copy2(str(current), str(final))
            current = final

        print_fn("")
        print_fn(f"Complete. Output: {current}")
        return current
    finally:
        print_fn("Cleaning up temp files...")
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_bulk(config: dict, input_dir: str | Path, output_dir: str | Path, print_fn=print) -> list[Path]:
    input_dir = Path(input_dir)
    files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in AUDIO_EXTS)
    if not files:
        print_fn(f"No audio files found in {input_dir}")
        return []

    print_fn(f"BULK MODE -- {len(files)} file(s) in {input_dir}")
    print_fn("")
    results = []
    for i, f in enumerate(files, start=1):
        print_fn(f"[BULK {i}/{len(files)}] {f.name}")
        try:
            results.append(run_chain(config, f, output_dir, print_fn=print_fn))
        except DegradeError as ex:
            print_fn(f"  ERROR: {ex} -- skipping.")
        print_fn("")
    print_fn(f"BULK MODE complete. {len(results)}/{len(files)} succeeded.")
    return results


# ---------------------------------------------------------------------------
# TUI entry point
# ---------------------------------------------------------------------------

def screen_degrade_audio(state: dict, console, _pick, _run_with_live_output, ROOT: Path) -> None:
    """Entry point called from tui.py."""
    configs_dir = ROOT / "utils" / "degrade"
    configs = list_configs(configs_dir)
    if not configs:
        console.clear()
        console.print(f"[red]No degrade configs found in {configs_dir}[/]")
        console.input("Press Enter to return.")
        return

    last_cfg = state.get("degrade", {}).get("last_config", "")
    start = next((i for i, c in enumerate(configs) if c.name == last_cfg), 0)
    items = [c.stem for c in configs]

    idx = _pick("Degrade Audio -- select config", items, hint="Enter=select  Esc=back", start=start)
    if idx is None:
        return
    cfg_path = configs[idx]
    state.setdefault("degrade", {})["last_config"] = cfg_path.name

    input_dir = ROOT / "input"
    output_dir = ROOT / "output"
    input_dir.mkdir(exist_ok=True)

    files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in AUDIO_EXTS)
    bulk_label = f"[ Process all {len(files)} file(s) in /input ]" if files else "[ Process all in /input (folder empty) ]"
    items = [bulk_label, "[ Enter custom file/folder path ]"] + [f.name for f in files]

    idx2 = _pick("Degrade Audio -- select input", items, hint="Enter=select  Esc=back")
    if idx2 is None:
        return

    console.clear()
    console.print(f"[bold cyan]Degrade Audio[/]  --  config: {cfg_path.stem}\n")

    lines = []
    def _print(*args):
        msg = " ".join(str(a) for a in args)
        lines.append(msg)
        console.print(msg)

    try:
        config = load_config(cfg_path)
        if idx2 == 0:
            if not files:
                console.print("[yellow]No files in /input.[/]")
            else:
                run_bulk(config, input_dir, output_dir, print_fn=_print)
        elif idx2 == 1:
            path = console.input("[cyan]Enter file or folder path:[/] ").strip().strip('"')
            if not path:
                return
            p = Path(path)
            if p.is_dir():
                run_bulk(config, p, output_dir, print_fn=_print)
            else:
                run_chain(config, p, output_dir, print_fn=_print)
        else:
            run_chain(config, files[idx2 - 2], output_dir, print_fn=_print)
    except DegradeError as ex:
        console.print(f"[red]Error: {ex}[/]")
    except Exception as ex:
        console.print(f"[red]Unexpected error: {ex}[/]")

    console.input("\n[dim]Press Enter to return to menu[/]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic audio degradation pipeline.")
    parser.add_argument("--config", required=True, help="Path to a degrade config JSON.")
    parser.add_argument("--input", required=True, help="Input audio file or folder (with --bulk).")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--bulk", action="store_true", help="Treat --input as a folder and process every audio file in it.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.bulk:
        run_bulk(config, args.input, args.output)
    else:
        run_chain(config, args.input, args.output)


if __name__ == "__main__":
    main()
