# UniverSR-mod

Fork of woongzip1/UniverSR (MIT, ICASSP 2026) made trainable and finetunable on consumer GPUs, with paired LQ/HQ data and MP3 (codec) restoration as the primary experiment. Sibling project: Emball/Apollo-mod (reference for TUI, metrics, checkpoint naming).

## Target
- Primary: RTX 2080 Ti 11 GB (Turing: fp16 AMP + GradScaler, no bf16), 16 GB system RAM.
- Modern GPUs (Ampere+): bf16 auto-selected.
- Dataset: 44.1 kHz source music. LQ and HQ are resampled once, offline, with the same filter to the model's 48 kHz and cached; training runs at 48 kHz.
- Pretrained base: `woongzip1/universr-audio` (48 kHz, n_fft 1024, hop 512, 512 bins). The sandbox cannot reach Hugging Face; the user downloads it locally and points the config at it.

## Design decisions
- Training rate is 48 kHz (the checkpoint's rate), a config value. Source files at other rates are resampled once during pair prep, never per step.
- Hard `sr_to_lr_bins` split is replaced by a three-zone soft split driven by per-sample cutoff Hz:
  clean (conditioning only) / transition (degraded, regenerated) / generated (above codec lowpass).
- Degraded full-band spectrum and a per-bin cutoff mask enter as extra conditioning channels. New input weights are zero-initialised so step 0 equals the pretrained model.
- Pairs: HQ -> LAME at source rate (mixed CBR/VBR) -> decode; encoder delay aligned; then LQ and HQ resampled identically to 48 kHz. Output may be resampled back to the source rate for delivery.
- Experiment arms, same held-out set: (1) zero-shot pretrained, (2) finetune original architecture with per-sample cutoffs, (3) finetune soft split + extra channels. Compare vs Apollo-mod on ViSQOL, LSD-high, HFNR, SI-SDR.
- Source content ends at 22.05 kHz, so the top bins of the 48 kHz spectrum are empty targets.

## Layout (upstream)
`train.py`, `evaluate.py`, `universr/{models/unet.py,flow/,trainer/trainer.py,utils/}`, `data/dataset.py`, `configs/config.yaml`, `scripts/`.

## Phases
0. Fork, AGENTS.md, VERSION. 
1. Config plumbing: remove `sr_to_lr_bins` use; cutoff in Hz; keep 48 kHz as config value.
2. Paired dataset: LQ/HQ folders or pairs list, aligned random crops, chunking, per-sample cutoff.
3. MP3 pair generator (`scripts/make_pairs.py`): encode, delay alignment, shared resample to 48 kHz, bitrate mix, cutoff estimation.
4. Soft-split model: extra cond channels, zero-init, loss over generated+transition region.
5. Trainer rewrite: AMP (fp16/bf16 auto), grad accumulation, activation checkpointing, EMA, grad clip, 8-bit optimizer option, torch.compile, step-based resume.
6. Finetune from pretrained: checkpoint loading with partial/shape-mismatch handling, freezing or LoRA.
7. Validation: rotating subset, metrics (ViSQOL, LSD-high, HFNR, SI-SDR), configurable best-checkpoint criterion.
8. Configs per VRAM tier (8/11/16 GB) and zero-shot/eval harness for the three arms.
9. TUI (train / inference / evaluate), `start.sh` + `start.bat`.
10. Optional: LoRA, cutoff detector for inference.

## Versioning
`VERSION` file, format MAJOR.MINOR.PATCH.MICRO. Commit message is the version only.
