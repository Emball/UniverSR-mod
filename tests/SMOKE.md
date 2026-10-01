# Smoke tests (run on the GPU machine)

Nothing below has been executed on real hardware. Run in order; stop at the first failure and report the log.
Config below: `configs/universr_stfl_new.yaml` (or `universr_mp3.yaml`). Data goes in `data/<exp.name>/{train,val}/{LQ,HQ}`.

1. **Setup.** Put `pytorch_model.bin` and `config.yaml` from `woongzip1/universr-audio` in `models/`. Run `universr.bat` (or `./universr.sh`): the TUI opens with the UniverSR-mod banner and lists the experiment configs, not `universr_pretrained`.
2. **CPU unit tests.** `pip install pytest`, then `python -m pytest tests -q --ignore=tests/test_train_e2e.py`. Expect all pass; `test_model_equivalence` is the step-0 check against the released U-Net.
3. **Real checkpoint loads.** `universr.bat inference --in_wav <any 44.1 kHz file> --out_wav out.wav --weights pretrained`. Expect a 48 kHz float WAV, no shape errors from the loader (the loader has only seen a random stand-in so far).
4. **Short training run.** Set `training.max_steps: 40` and `val_check_interval: 20`, then `universr.bat train --conf_dir configs/universr_stfl_new.yaml`. Check:
   - chunking and cache creation, then the baseline pass (arm A) with ViSQOL numbers
   - log says `fp16-mixed` on the 2080 Ti, loss finite (no NaN with the GradScaler)
   - validation fires at steps 20 and 40; checkpoint names `step=000020-sisdr=...-visqol=...-hfnr=...`; `best_model.pth` appears
   - peak VRAM at `batch_size: 8` with `grad_checkpoint: true` (`nvidia-smi`); on OOM try batch 4, or `grad_accum_steps: 2`
   - speed with `grad_checkpoint: false` if VRAM has headroom
5. **Ctrl+C and resume.** Start training, Ctrl+C after a few steps: a `step=N.ckpt` is saved. Rerun with `--resume`: continues from step N, not from 0 (this is the upstream bug the fork fixes). Create `runs/<name>/PAUSED` mid-run: training suspends; delete it: continues.
6. **Freeze setting.** The `[freeze]` log line should report the pretrained backbone frozen and a small trainable share. If loss does not move after a few hundred steps, switch `freeze` to the commented alternative in the config.
7. **Evaluate.** `universr.bat evaluate --conf_dir configs/universr_stfl_new.yaml --baseline`: a table with the zero-shot baseline row and each checkpoint on the same clips.
8. **Inference from the TUI.** Run restoration with a `.ckpt` and with `best_model.pth`, including the new options menu (fixed cutoff, match input rate) and one ensemble preset. Mid-training inference (Ctrl+I) should also run.
9. **Optional.** `optimizer.type: adamw_8bit` and `gefen`, to see that they start and step.
