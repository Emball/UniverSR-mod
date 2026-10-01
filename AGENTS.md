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
- Level handling matches upstream (song-level peak, then crop): chunks are cached with the pair-shared song peak at `datas.cache_peak_dbfs` (-1). At load, a relative gain puts the song peak at a random -6..-1 dBFS (train) or -3 dBFS (val); crops are never renormalised, so quiet passages stay quiet. Inference restores the original level.
- Stereo: as Apollo-mod. Training sees single channels via the `stereo_alternation` augmentation (L first half of a song, R second half; skipped when `mid_side_isolation` fires). Inference processes L and R independently and re-stacks them; shared noise across channels is an optional flag, off by default.
- Conditioning generalised to per-sample bandwidth: embedding interpolated between anchor rows (first four = pretrained rows), masked mean over valid bins.
- Repair path (arm C): bin-aligned degraded spectrum + validity mask as extra `init_conv` input channels over the generated region, zero-initialised. Generated region can start at bin 80 (default) or 0.
- Assembly: arm B keeps real bins up to the cutoff; arm C takes generated bins from the generated-region start.
- Flow-matching loss supports per-Hz band weights (as Apollo `band_weight_*`).
- GAN discriminator and its losses are not ported. Apollo optimizer factory (adamw, 8-bit, gefen, gefen_muon), grad accumulation, checkpointing, RAM watchdog, augmentations, spectral-merge ensemble are.
- Arms on the same held-out set: A zero-shot pretrained (baseline val pass; real bins kept below cutoff), B cutoff-aware finetune, C B + aligned channels. Metrics: ViSQOL, LSD-high, HFNR, SI-SDR; SI-SDR is expected to trail for a generative model.

## Phases
0. Fork, AGENTS.md, VERSION.
1. Restructure to Apollo layout (`core/`, `utils/`, `configs/`, `models/`, `runs/`); port TUI, degrade/align tools, launchers; remove upstream-only entry points. Done: upstream train/evaluate/dataset kept in `core/_upstream/` as reference until phases 2-5 replace them; TUI is a verbatim copy pending phase 8 deltas.
2. Data: port paired datamodule and chunk/val-clip prep at 48 kHz; mono crops; level handling; augmentations (live + cached). Done: `core/{audio_io,augment,cutoff,data_prep,paired_datamodule}.py`. Cache keys include sample rate and alignment. Delays (`align_data`) are in samples of the file being trimmed, applied before resampling. All resampling goes through `audio_io.resample`. Per-sample cutoff comes from `datas.cutoff_hz` (`auto` | Hz | [min,max]); `auto` detects the spectral drop on the augmented LQ, so live `noise` augmentation defeats it.
3. Backend: cutoff-aware conditioning, aligned channels, region config, pretrained loader with shape reconciliation; CPU step-0 equivalence test against original U-Net. Done: `core/universr/models/{unet_mod,loader}.py`, `tests/test_model_equivalence.py` (CPU; matches the original U-Net to ~4e-7 at step 0 for arms B and C, conditional and unconditional). API: `forward(x, t, y, cutoff_bins)`; `y` is the LQ spec from bin 0 (arm C also reads bins from `gen_start_bin` up); `cutoff_bins` is per sample; `y=None` is unconditional. Bandwidth anchors default to (80,128,170,256,341,405,470,512); the pretrained 4-row table is interpolated onto them. Per-block checkpointing is in the model (`grad_checkpoint`). In mixed-cutoff batches, samples narrower than the batch max differ slightly from a standalone run (zero vs reflect padding at the cutoff edge, up to ~4% of output scale at 80 bins); uniform cutoffs or per-sample runs are exact.
4. LightningModule: flow-matching step (fp32 STFT, AMP), band weights, accumulation, clipping, block checkpointing, ODE validation with locking/rotation, baseline cache. Done: `core/universr/{system,sampling,metrics}.py`, `flow/loss.py`, tests in `tests/` (CPU, Lightning end to end). `UniverSRSystem` uses automatic optimization: accumulation, clipping and precision are Trainer args; the optimizer and a step scheduler are passed in; the network is `audio_model`. It logs `train_loss`, `sisdr` (real SI-SDR, higher is better), `hfnr`, `visqol` (-1 if unavailable), `lsd_high`, `val_cfm`. Validation runs the ODE sampler (midpoint, CFG, per-clip seed) on every clip of the `val_songs` songs in the active window; the window rotates every `val_rotate_every` steps (`auto` = max_steps / windows); song order, window index and EMA are saved in checkpoints. `ema_decay` > 0 keeps an EMA that is swapped in for validation and stored under `ema` in checkpoints. Per-band loss weights: `band_weights: [{lo_hz, hi_hz, weight}]`. Preview WAVs go to `val_audio_dir/step_N`. The baseline pass is `trainer.validate` on the pretrained weights; caching it is train.py's job (phase 5).
5. `train.py` port: run isolation, StepPrinter, PAUSED, Ctrl+C save, optimizer factory, precision auto, weights lookup, resume.
6. Inference and evaluate: ODE OLA, level restore, ensemble, stereo mode, cutoff detector with override.
7. Configs per VRAM tier (8/11/16 GB), memory profiler utility, arm configs.
8. TUI deltas, launchers, smoke tests.

## Upstream training distribution (keep data matched)
- Samples are random 32767-sample crops (~0.68 s, `num_samples`) of whole files, one per file per epoch, mean-downmixed to mono. We crop from cached 3 s chunks at 50% overlap (every crop fits inside some chunk) and take a single channel via `stereo_alternation`.
- Upstream draws one bandwidth per batch; we use a per-sample cutoff in Hz.
- Upstream val is the first 5 s of each file; ours is two 10 s highest-RMS clips per song (ViSQOL-compliant).

## STFT facts
- The transform's Hann window is symmetric (zero at both ends). Inverting a modified spectrum divides by a near-zero envelope in the last <hop samples unless the length is a multiple of the hop, so every inversion path uses `to_spec(..., pad_to_hop=True)` and `to_wave` (natural length, then crop). Training keeps upstream framing (no pad; 32767 samples = 64 frames).
- Sampling uses `sampling.py` (own midpoint solver); `torchdiffeq` is only needed by `core/_upstream/`. WAVs are written with soundfile (`audio_io.save_wav_f32`) because torchaudio >= 2.9 needs TorchCodec for I/O.

## Versioning
`VERSION` file, MAJOR.MINOR.PATCH.MICRO. Commit message is the version only.
