"""
inference.py -- UniverSR-mod restoration

Usage:
    # Best ranked checkpoint of the experiment in a config
    python core/inference.py --in_wav in.wav --out_wav out.wav --conf_dir configs/universr_stfl2.yaml

    # Explicit weights: Lightning .ckpt (needs --conf_dir), exported best_model.pth, or pretrained .bin
    python core/inference.py --in_wav in.wav --out_wav out.wav --weights runs/x/ts/best_model.pth

    # Zero-shot pretrained model
    python core/inference.py --in_wav in.wav --out_wav out.wav --weights pretrained

    # Spectral merge with the original (Apollo-mod semantics) and a second checkpoint
    python core/inference.py --in_wav in.wav --out_wav out.wav --weights a.pth --low_end_preserve
    python core/inference.py --in_wav in.wav --out_wav out.wav --weights a.pth --ensemble '[{"lo":0,"hi":700,"mode":"max_fft"}]' \\
        --aux_weights b.pth --aux_ensemble '[{"lo":8000,"hi":24000,"mode":"avg","weight":0.5}]'
"""
import argparse
import logging
import os
import sys

CORE = os.path.dirname(os.path.abspath(__file__))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from audio_io import save_wav_f32  # noqa: E402
from ensemble import low_end_bands, parse_bands  # noqa: E402
from restorer import load_input, load_restorer, restore_audio  # noqa: E402

log = logging.getLogger("universr.inference")


def _progress():
    last = {"tag": None, "pct": -1}

    def cb(tag, done, total):
        pct = int(100 * done / max(1, total))
        if tag != last["tag"] or pct != last["pct"]:
            last["tag"], last["pct"] = tag, pct
            print(f"\r[inference] {tag}  {pct:3d}%", end="\n" if done >= total else "", flush=True)
    return cb


def run(in_wav, out_wav, weights=None, conf_dir=None, device="auto", precision="auto", chunk_sec=6.0,
        overlap_sec=0.5, chunked=True, cutoff_hz="auto", steps=None, guidance=None, seed=1234,
        shared_noise=False, match_input_sr=False, ensemble=None, low_end_preserve=False, low_end_hz=700.0,
        aux_weights=None, aux_conf_dir=None, aux_ensemble=None):
    bands = parse_bands(ensemble)
    if bands is None and low_end_preserve:
        bands = low_end_bands(low_end_hz)
    if bands:
        log.info("spectral merge bands: %s", bands)

    rs = load_restorer(weights, conf_dir, device, precision)
    aux = load_restorer(aux_weights, aux_conf_dir, device, precision) if aux_weights else None

    wav, sr_in = load_input(in_wav)
    log.info("input: %s (%.1f s, %d ch, %d Hz)", in_wav, wav.shape[-1] / sr_in, wav.shape[0], sr_in)
    out, out_sr = restore_audio(
        rs, wav, sr_in, cutoff=cutoff_hz, chunk_sec=chunk_sec if chunked else None, overlap_sec=overlap_sec,
        steps=steps, guidance=guidance, seed=seed, shared_noise=shared_noise, bands=bands, aux=aux,
        aux_bands=parse_bands(aux_ensemble), match_input_sr=match_input_sr, progress=_progress())

    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    save_wav_f32(out, out_wav, out_sr)
    log.info("saved -> %s (%d Hz)", out_wav, out_sr)


def main():
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    ap = argparse.ArgumentParser(description="UniverSR-mod restoration")
    ap.add_argument("--in_wav", required=True, help="input audio file (any format ffmpeg can decode)")
    ap.add_argument("--out_wav", required=True, help="output path (32-bit float WAV)")
    ap.add_argument("--weights", default=None,
                    help=".ckpt (needs --conf_dir), exported .pth, pretrained .bin, or 'pretrained'; "
                         "omit to auto-select the best checkpoint of --conf_dir's experiment")
    ap.add_argument("--conf_dir", default=None, help="training yaml (model, transform, system settings)")
    ap.add_argument("--chunk_sec", type=float, default=6.0, help="chunk length in seconds")
    ap.add_argument("--overlap_sec", type=float, default=0.5, help="chunk overlap in seconds")
    ap.add_argument("--no_chunked", action="store_true", help="run each channel in one pass (may run out of VRAM)")
    ap.add_argument("--device", default="auto", help="'auto', 'cuda', 'cpu', 'cuda:1', ...")
    ap.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"],
                    help="auto: bf16 on Ampere+, fp16 on Turing, fp32 on CPU")
    ap.add_argument("--cutoff_hz", default=None,
                    help="codec lowpass in Hz, or 'auto' to detect it; default: datas.cutoff_hz from the config, else auto")
    ap.add_argument("--steps", type=int, default=None, help="ODE steps (default: from config, else 4)")
    ap.add_argument("--guidance", type=float, default=None, help="CFG scale (default: from config, else 1.5)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shared_noise", action="store_true", help="use the same noise for L and R")
    ap.add_argument("--match_input_sr", action="store_true", help="resample the output back to the input's rate")
    ap.add_argument("--low_end_preserve", action="store_true",
                    help="max_fft blend with the original below --low_end_hz")
    ap.add_argument("--low_end_hz", type=float, default=700.0)
    ap.add_argument("--ensemble", default=None, help="JSON band specs: lo, hi, mode, weight")
    ap.add_argument("--aux_weights", default=None, help="second checkpoint for a dual-checkpoint ensemble")
    ap.add_argument("--aux_conf_dir", default=None, help="config for the aux checkpoint")
    ap.add_argument("--aux_ensemble", default=None, help="JSON band specs for the aux blend (default: --ensemble)")
    a = ap.parse_args()

    cutoff = a.cutoff_hz
    if cutoff is None:
        cutoff = "auto"
        try:
            import yaml
            c = yaml.safe_load(open(a.conf_dir)).get("datas", {}).get("cutoff_hz", "auto")
            if isinstance(c, (int, float)) and not isinstance(c, bool):
                cutoff = c
        except Exception:
            pass
    if cutoff != "auto":
        cutoff = float(cutoff)
    run(a.in_wav, a.out_wav, weights=a.weights, conf_dir=a.conf_dir, device=a.device, precision=a.precision,
        chunk_sec=a.chunk_sec, overlap_sec=a.overlap_sec, chunked=not a.no_chunked, cutoff_hz=cutoff,
        steps=a.steps, guidance=a.guidance, seed=a.seed, shared_noise=a.shared_noise,
        match_input_sr=a.match_input_sr, ensemble=a.ensemble, low_end_preserve=a.low_end_preserve,
        low_end_hz=a.low_end_hz, aux_weights=a.aux_weights, aux_conf_dir=a.aux_conf_dir, aux_ensemble=a.aux_ensemble)


if __name__ == "__main__":
    main()
