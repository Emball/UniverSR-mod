# UniverSR-mod

Fork of woongzip1/UniverSR (MIT, ICASSP 2026) made trainable and finetunable on consumer GPUs, with paired LQ/HQ data and MP3 (codec) restoration as the primary experiment. Sibling project: Emball/Apollo-mod (reference for TUI, metrics, checkpoint naming).

## Target
- Primary: RTX 2080 Ti 11 GB (Turing: fp16 AMP + GradScaler, no bf16), 16 GB system RAM.
- Modern GPUs (Ampere+): bf16 auto-selected.
- Dataset: 44.1 kHz stereo/mono music, native rate end to end. No resampling anywhere.
- Pretrained base: `woongzip1/universr-audio` (48 kHz, n_fft 1024, hop 512, 512 bins). The sandbox cannot reach Hugging Face; the user downloads it locally and points the config at it.

## Design decisions
- Sample rate is a config value. A run trains at one rate; other-rate files are skipped with a logged warning.
- Hard `sr_to_lr_bins` split is replaced by a three-zone soft split driven by per-sample cutoff Hz:
  clean (conditioning only) / transition (degraded, regenerated) / generated (above codec lowpass).
- Degraded full-band spectrum and a per-bin cutoff mask enter as extra conditioning channels. New input weights are zero-initialised so step 0 equals the pretrained model.
- Pairs: HQ -> LAME (mixed CBR/VBR) -> decode at the same rate; encoder delay is aligned before cropping.
- Experiment arms, same held-out set: (1) zero-shot pretrained, (2) finetune original architecture with per-sample cutoffs, (3) finetune soft split + extra channels. Compare vs Apollo-mod on ViSQOL, LSD-high, HFNR, SI-SDR.
- Bin-spacing shift (43.07 vs 46.875 Hz) is absorbed by finetuning. Optional later experiment: rate-fade curriculum.

## Layout (upstream)
`train.py`, `evaluate.py`, `universr/{models/unet.py,flow/,trainer/trainer.py,utils/}`, `data/dataset.py`, `configs/config.yaml`, `scripts/`.

## Phases
0. Fork, AGENTS.md, VERSION. 
1. Config + sample-rate plumbing: remove hardcoded 48k / `sr_to_lr_bins` use; cutoff in Hz.
2. Paired dataset: LQ/HQ folders or pairs list, aligned random crops, chunking, per-sample cutoff.
3. MP3 pair generator (`scripts/make_pairs.py`), delay alignment, bitrate mix, cutoff estimation.
4. Soft-split model: extra cond channels, zero-init, loss over generated+transition region.
5. Trainer rewrite: AMP (fp16/bf16 auto), grad accumulation, activation checkpointing, EMA, grad clip, 8-bit optimizer option, torch.compile, step-based resume.
6. Finetune from pretrained: checkpoint loading with partial/shape-mismatch handling, freezing or LoRA.
7. Validation: rotating subset, metrics (ViSQOL, LSD-high, HFNR, SI-SDR), configurable best-checkpoint criterion.
8. Configs per VRAM tier (8/11/16 GB) and zero-shot/eval harness for the three arms.
9. TUI (train / inference / evaluate), `start.sh` + `start.bat`.
10. Optional: rate-fade curriculum, LoRA, cutoff detector for inference.

## Versioning
`VERSION` file, format MAJOR.MINOR.PATCH.MICRO. Commit message is the version only.
