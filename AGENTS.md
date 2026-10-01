# UniverSR-mod

Fork of woongzip1/UniverSR (MIT, ICASSP 2026): trainable and finetunable on consumer GPUs from user-supplied paired LQ/HQ audio. Same user-facing shell as Emball/Apollo-mod (TUI, configs, run folders, checkpoint names, validation), different backend (flow-matching ConvNeXt U-Net instead of Apollo). Primary experiment: MP3/codec restoration finetuned from the pretrained model.

## Target
- RTX 2080 Ti 11 GB (Turing: fp16 AMP, no bf16), 16 GB system RAM. Ampere+ auto-selects bf16.
- Pretrained base: `woongzip1/universr-audio` (`config.yaml` + `pytorch_model.bin`). Sandbox cannot reach Hugging Face; place files in `models/`.
- The user supplies all data. No synthetic LQ generation in training. `utils/degrade_audio.py` stays as a manual tool.

## Pretrained model facts
- Mono, 48 kHz, STFT n_fft 1024 / hop 512, 512 bins (Nyquist dropped), amplitude-compressed (alpha 0.2) complex STFT as 2 real channels.
- Condition path: low bins -> FiLM(freq pos-emb) -> ConvNeXt blocks -> mean over frequency -> one vector per frame. The U-Net never sees bin-aligned condition.
- Generated region is always bins 80-511 (432). Upstream discards the part overlapping the condition at assembly.
- Input bandwidth is a 4-row table (8/12/16/24 kHz -> 80/128/170/256 bins) with a 4-row embedding. MP3 lowpass (16-20 kHz) is beyond the trained range.
- Trained on peak-normalised audio (random -1 to -6 dBFS), 32767-sample crops; 10% condition dropout for CFG; sampling 4-step midpoint, guidance 1.5.

## Apollo-mod contract (the TUI depends on it)
- CLI: `core/train.py --conf_dir X [--resume] [--weights_path P]`, `core/inference.py --in_wav --out_wav --weights --conf_dir`, `core/evaluate.py --conf_dir`.
- `configs/*.yaml` with `exp.name`/`exp.dir`; runs in `runs/<name>/<timestamp>/checkpoints/*.ckpt`.
- Checkpoint name `step={:06d}-sisdr={:.3f}-visqol={:.3f}-hfnr={:.3f}`; best = `(visqol, sisdr, -hfnr)`.
- `PAUSED` sentinel in `runs/<name>/` suspends training; Ctrl+C saves a checkpoint.
- Data: `data/<name>/{train,val}/{LQ,HQ}` -> chunk cache under `usr/cache/chunks/<key>/`; two 10 s highest-RMS val clips per song, locked and rotated.

## Design decisions
- Pairs are resampled once at chunk-cache time to 48 kHz (same filter for LQ and HQ); `_SR` is a config value. Output can be resampled back for delivery.
- Lightning + Hydra shell as in Apollo-mod; backend is `core/universr/` (replaces `core/look2hear/`).
- Level handling: pair-shared peak normalisation (train: random -6..-1 dBFS; val/inference: -3 dBFS, restored after).
- Stereo: train on single channels (mono model); inference processes channels independently with optional shared noise. Tested in the arm comparison.
- Conditioning generalised to per-sample bandwidth: embedding interpolated between anchor rows (first four = pretrained rows), masked mean over valid bins.
- Repair path (arm C): bin-aligned degraded spectrum + validity mask as extra `init_conv` input channels over the generated region, zero-initialised. Generated region can start at bin 80 (default) or 0.
- Assembly: arm B keeps real bins up to the cutoff; arm C takes generated bins from the generated-region start.
- Flow-matching loss supports per-Hz band weights (as Apollo `band_weight_*`).
- GAN discriminator and its losses are not ported. Apollo optimizer factory (adamw, 8-bit, gefen, gefen_muon), grad accumulation, checkpointing, RAM watchdog, augmentations, spectral-merge ensemble are.
- Arms on the same held-out set: A zero-shot pretrained (baseline val pass; real bins kept below cutoff), B cutoff-aware finetune, C B + aligned channels. Metrics: ViSQOL, LSD-high, HFNR, SI-SDR; SI-SDR is expected to trail for a generative model.

## Phases
0. Fork, AGENTS.md, VERSION.
1. Restructure to Apollo layout (`core/`, `utils/`, `configs/`, `models/`, `runs/`); port TUI, degrade/align tools, launchers; remove upstream-only entry points.
2. Data: port paired datamodule and chunk/val-clip prep at 48 kHz; mono crops; level handling; augmentations (live + cached).
3. Backend: cutoff-aware conditioning, aligned channels, region config, pretrained loader with shape reconciliation; CPU step-0 equivalence test against original U-Net.
4. LightningModule: flow-matching step (fp32 STFT, AMP), band weights, accumulation, clipping, block checkpointing, ODE validation with locking/rotation, baseline cache.
5. `train.py` port: run isolation, StepPrinter, PAUSED, Ctrl+C save, optimizer factory, precision auto, weights lookup, resume.
6. Inference and evaluate: ODE OLA, level restore, ensemble, stereo mode, cutoff detector with override.
7. Configs per VRAM tier (8/11/16 GB), memory profiler utility, arm configs.
8. TUI deltas, launchers, smoke tests.

## Versioning
`VERSION` file, MAJOR.MINOR.PATCH.MICRO. Commit message is the version only.
