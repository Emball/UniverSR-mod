# Smoke tests (run on the GPU machine)

Run on the 2080 Ti so far: setup and the TUI, chunking, the baseline pass, fp16 training and validation (steps 1 and 4 in part). Everything else has not been executed. Run in order; stop at the first failure and report the log.
Config below: `configs/itunes_mp3.yaml` (or `universr_stfl_new.yaml`). Data goes in `data/<exp.name>/{train,val}/{LQ,HQ}`.

1. **Setup.** Put `pytorch_model.bin` and `config.yaml` from `woongzip1/universr-audio` in `models/`. Run `universr.bat` (or `./universr.sh`): the TUI opens with the UniverSR-mod banner and lists the experiment configs, not `universr_pretrained`.
2. **CPU unit tests.** `pip install pytest`, then `python -m pytest tests -q --ignore=tests/test_train_e2e.py`. Expect all pass; `test_model_equivalence` is the step-0 check against the released U-Net.
3. **Real checkpoint loads.** `universr.bat inference --in_wav <any 44.1 kHz file> --out_wav out.wav --weights pretrained`. Expect a 48 kHz float WAV (the pretrained config), no shape errors from the loader (the loader has only seen a random stand-in so far).
3b. **Speed and fp16 check.** `python tests/bench_gpu.py --conf_dir configs/universr_stfl_new.yaml`. Compares model variants (ms/step, peak VRAM) and, with the pretrained weights, names the first module that goes non-finite in fp16.
4. **Short training run.** Set `training.max_steps: 40` and `val_check_interval: 20`, then `universr.bat train --conf_dir configs/universr_stfl_new.yaml`. Check:
   - chunking and cache creation, then the baseline pass (arm A) with ViSQOL numbers
   - log says `fp16-mixed` on the 2080 Ti, loss finite (no NaN with the GradScaler)
   - validation fires at steps 20 and 40; checkpoint names `step=000020-sisdr=...-visqol=...-hfnr=...`; `best_model.pth` appears
   - peak VRAM at `batch_size: 8` with `grad_checkpoint: true` (`nvidia-smi`); on OOM try batch 4, or `grad_accum_steps: 2`
   - speed with `grad_checkpoint: false` if VRAM has headroom
5. **Ctrl+C and resume.** Start training, Ctrl+C after a few steps: a `step=N.ckpt` is saved. Rerun with `--resume`: continues from step N, not from 0 (this is the upstream bug the fork fixes). Create `runs/<name>/PAUSED` mid-run: training suspends; delete it: continues.
6. **Freeze setting.** The shipped configs set `freeze: []`. With a prefix list set, the `[freeze]` log line reports the frozen share of parameters.
7. **Evaluate.** `universr.bat evaluate --conf_dir configs/universr_stfl_new.yaml --baseline`: a table with the zero-shot baseline row and each checkpoint on the same clips.
8. **Inference from the TUI.** Run restoration with a `.ckpt` and with `best_model.pth`, including the new options menu (fixed cutoff, match input rate) and one ensemble preset. Mid-training inference (Ctrl+I) should also run.
9. **Optional.** `optimizer.type: adamw_8bit` and `gefen`, to see that they start and step.
10. **Native 44.1 kHz run.** Move any older 48 kHz run folder out of `runs/itunes_mp3/`, set `weights_path` in `configs/itunes_mp3.yaml` to a checkpoint from it, then train. Check: the loader log shows `freq_pos_enc.pe: sliced (512, 384) -> (480, 384)` and no skipped tensors; chunking reports `at 44100 Hz`; the baseline pass runs with real ViSQOL numbers; the first validations recover to the 48 kHz run's trend; VRAM at `batch_size: 6` stays near the 48 kHz run's.
