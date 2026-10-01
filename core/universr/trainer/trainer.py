import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime

import soundfile as sf
import torch
import torch.nn as nn
import wandb
from torch.optim import lr_scheduler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from universr.flow.loss import flow_matching_loss
from universr.flow.solver import CFGVectorFieldODE, TorchDiffeqSolver, VectorFieldODE
from universr.utils.utils import draw_spec, t2n

try:
    from torch_peaq import PEAQBasic
    PEAQ_AVAILABLE = True
except ImportError:
    print("Warning: torch_peaq not available. 2f-model metrics will be disabled.")
    PEAQ_AVAILABLE = False


MiB = 1024 ** 2


def model_size_b(model: nn.Module) -> int:
    """Returns model size in bytes."""
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


class Trainer(ABC):
    """Abstract base class for training."""

    def __init__(self, model, train_loader, val_loader, device, logger):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.start_epoch = 1
        self.best_loss = float('inf')
        self.optimizer = None
        self.scheduler = None
        self.logger = logger

    @abstractmethod
    def _train_step(self, **kwargs) -> torch.Tensor:
        pass

    def _val_step(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, config):
        return torch.optim.Adam(self.model.parameters(), **config)

    def get_scheduler(self, optimizer, config):
        scheduler_type = config.get('type', 'CosineLR')
        scheduler_args = config.get('init_args', {})
        if scheduler_type == 'ExponentialLR':
            return lr_scheduler.ExponentialLR(optimizer, **scheduler_args)
        elif scheduler_type == 'CosineLR':
            return get_cosine_schedule_with_warmup(optimizer=optimizer, **scheduler_args)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    # ------------------------------------------------------------------ #
    #  Checkpointing
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, epoch, is_best, save_dir, filename=None):
        os.makedirs(save_dir, exist_ok=True)
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        ckpt_path = os.path.join(save_dir, filename or 'recent.pth')
        torch.save(state, ckpt_path)
        print(f"Checkpoint saved at epoch {epoch}: {ckpt_path}")

        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pth')
            shutil.copyfile(ckpt_path, best_path)
            print(f"Best model updated at epoch {epoch} (loss={self.best_loss:.6f})")

    def load_checkpoint(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            print(f"Checkpoint not found at {ckpt_path}. Starting from scratch.")
            return
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be initialized before loading a checkpoint.")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Training checkpoint loaded from {ckpt_path}. Resuming from epoch {self.start_epoch}.")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"Model weights loaded from {ckpt_path}. Starting training from epoch 1.")

    @staticmethod
    def load_model_for_inference(model, ckpt_path, device='cuda'):
        model.to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"Model loaded from {ckpt_path}")
        return model

    # ------------------------------------------------------------------ #
    #  Validation / Train loops
    # ------------------------------------------------------------------ #
    def validate(self, global_step, val_idx, ode_steps=4, val_max_sec=5):
        self.model.eval()
        total_val_loss = 0.0
        total_metric = 0.0
        cnt = 0
        val_pbar = tqdm(self.val_loader, desc='Validating...', dynamic_ncols=True)

        with torch.no_grad():
            for idx, batch in enumerate(val_pbar):
                outdict = self._val_step(batch, idx, val_idx,
                                         ode_steps=ode_steps, val_max_sec=val_max_sec)
                loss = outdict['loss']
                total_val_loss += loss.item()

                if outdict['lsd_high'] is not None:
                    total_metric += outdict['lsd_high']
                    cnt += 1
                val_pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})

                if outdict['log_payload']:
                    self.logger.log(outdict['log_payload'], step=global_step)

        avg_val_loss = total_val_loss / len(self.val_loader)
        avg_metric = total_metric / cnt if cnt > 0 else 0
        self.logger.log({"val/loss": avg_val_loss, "val/lsd_high": avg_metric}, step=global_step)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.train()
        return {"loss": avg_val_loss, "lsd_high": avg_metric}

    def train(self,
              num_epochs,
              max_steps=500000,
              optimizer_config=None,
              scheduler_config=None,
              ckpt_save_dir='ckpts',
              ckpt_load_path=None,
              log_step_interval=100,
              val_step_interval=5000,
              num_val_log_samples=10,
              val_ode_steps=4,
              val_max_sec=5,
              **kwargs):
        self.log_step_interval = log_step_interval
        total_val_batches = len(self.val_loader)
        val_idx = set(torch.linspace(0, total_val_batches - 1, num_val_log_samples).long().tolist())

        print(f'Training model with size: {model_size_b(self.model) / MiB:.3f} MiB')
        self.model.to(self.device)
        self.optimizer = self.get_optimizer(optimizer_config)
        if scheduler_config:
            self.scheduler = self.get_scheduler(self.optimizer, scheduler_config)
        if ckpt_load_path:
            self.load_checkpoint(ckpt_load_path)

        global_step = (self.start_epoch - 1) * len(self.train_loader)
        self.model.train()
        print(f"--- Starting training from epoch {self.start_epoch} ---")

        for epoch_idx in range(self.start_epoch, num_epochs + 1):
            epoch_pbar = tqdm(self.train_loader,
                              desc=f'Epoch {epoch_idx}/{num_epochs}',
                              dynamic_ncols=True, leave=True)
            total_epoch_loss = 0.0

            for batch_idx, batch in enumerate(epoch_pbar):
                global_step += 1

                self.optimizer.zero_grad()
                loss, loss_dict = self._train_step(batch, global_step)
                loss.backward()
                self.optimizer.step()
                if scheduler_config:
                    self.scheduler.step()

                total_epoch_loss += loss.item()
                epoch_pbar.set_postfix({'loss': f'{loss.item():.6f}'})

                # Step logging
                if global_step % log_step_interval == 0:
                    self.logger.log({"model/loss": loss.item()}, step=global_step)
                    self.logger.log({"charts/lr-adam": self.optimizer.param_groups[0]['lr']}, step=global_step)
                    self.logger.log({f"model/{k}": v.item() for k, v in loss_dict.items()}, step=global_step)

                # Validation
                if global_step % val_step_interval == 0:
                    val_results = self.validate(global_step, val_idx,
                                                ode_steps=val_ode_steps, val_max_sec=val_max_sec)
                    avg_val_loss = val_results['loss']
                    print(f'\nStep {global_step} | Val Loss: {avg_val_loss:.6f}, '
                          f'Val LSD-high: {val_results["lsd_high"]:.4f}\n')

                    is_best = avg_val_loss < self.best_loss
                    if is_best:
                        self.best_loss = avg_val_loss
                    self.save_checkpoint(epoch=epoch_idx, is_best=is_best, save_dir=ckpt_save_dir)

                if global_step >= max_steps:
                    print(f'\nReached max_steps ({max_steps}). Finishing training.')
                    self.save_checkpoint(epoch=epoch_idx, is_best=False, save_dir=ckpt_save_dir,
                                         filename=f'step_{global_step}.pth')
                    return

            avg_epoch_loss = total_epoch_loss / len(self.train_loader)
            print(f'Epoch {epoch_idx} completed. Average Loss: {avg_epoch_loss:.6f}')
            self.logger.log({"model/epoch_loss": avg_epoch_loss, "charts/epoch": epoch_idx}, step=global_step)

        self.model.eval()
        print('Training finished!')


# ====================================================================== #
#  STFTTrainer
# ====================================================================== #

class STFTTrainer(Trainer):
    """Flow-matching trainer operating in the STFT domain."""

    def __init__(self, model, path, transform, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path
        self.transform = transform

    # ------------------------------------------------------------------ #
    #  Spectral pre/post-processing
    # ------------------------------------------------------------------ #
    def _preprocess(self, waveform):
        """waveform [B,C,T] -> real-valued STFT [B,2,F,T]"""
        spec = self.transform(waveform)
        real = torch.view_as_real(spec.squeeze(1))
        real = real.permute(0, 3, 1, 2)
        return real[:, :, :-1, :]

    def _postprocess(self, spec):
        """real-valued STFT [B,2,F,T] -> waveform [B,T]"""
        spec = torch.nn.functional.pad(spec, pad=[0, 0, 0, 1], value=0)
        spec = spec.permute(0, 2, 3, 1).contiguous()
        spec = torch.view_as_complex(spec)
        return self.transform.invert(spec)

    def _get_freq_bins(self, sr_value):
        """Return (lr_bin_count, hf_start_bin) for a given sampling rate."""
        lr_bin_count = self.model.sr_to_lr_bins[sr_value]
        hf_start_bin = self.model.total_freq_bins - self.model.hr_freq_bins
        return lr_bin_count, hf_start_bin

    def _split_spectrum(self, Y, Z, lr_bin_count, hf_start_bin):
        """Split full spectra into LR condition and HR target regions."""
        Y_lr = Y[:, :, :lr_bin_count, :]
        Y_hr = Y[:, :, hf_start_bin:, :]
        Z_hr = Z[:, :, hf_start_bin:, :]
        return Y_lr, Y_hr, Z_hr

    def _assemble_fullband(self, Y_lr, x1_hr, lr_bin_count, hf_start_bin):
        """Concatenate LR condition with generated HR to form fullband spectrum."""
        slice_start = max(0, lr_bin_count - hf_start_bin)
        return torch.cat([Y_lr, x1_hr[:, :, slice_start:, :]], dim=2)

    # ------------------------------------------------------------------ #
    #  ODE inference
    # ------------------------------------------------------------------ #
    def _run_ode(self, Y_lr, Y_hr, sr_values, ode_steps, guidance_scale=None):
        """Run ODE sampling in the HR spectral region."""
        if guidance_scale is not None and guidance_scale != 0:
            ode = CFGVectorFieldODE(net=self.model, guidance_scale=float(guidance_scale))
        else:
            ode = VectorFieldODE(net=self.model)
        solver = TorchDiffeqSolver(ode, method='midpoint')
        ts = torch.linspace(0, 1, int(ode_steps) + 1, device=self.device)
        x0 = self.path.sample_source(Y_hr)
        return solver.simulate(x0, ts, y=Y_lr, sr_values=sr_values)

    def _synthesize_waveform(self, Y_lr, Y_hr, sr_values, lr_bin_count, hf_start_bin,
                             ode_steps, guidance_scale=None):
        """Full pipeline: ODE sampling -> spectral assembly -> waveform."""
        x1_hr = self._run_ode(Y_lr, Y_hr, sr_values, ode_steps, guidance_scale)
        x1_full = self._assemble_fullband(Y_lr, x1_hr, lr_bin_count, hf_start_bin)
        return self._postprocess(x1_full)

    # ------------------------------------------------------------------ #
    #  Metrics
    # ------------------------------------------------------------------ #
    def _stft_magnitude(self, audio, n_fft=1024, hop_length=256):
        """Compute STFT magnitude for metric calculations."""
        window = torch.hann_window(n_fft).to(audio.device)
        return torch.abs(torch.stft(audio, n_fft, hop_length, window=window, return_complex=True))

    def _compute_lsd(self, pred, target, sr_khz):
        """
        Compute Log-Spectral Distance with frequency band separation.
        Cutoff bin is derived from model config (sr_to_lr_bins).
        Returns: (lsd_total, lsd_high, lsd_low)
        """
        bin_idx = self.model.sr_to_lr_bins[sr_khz]

        sp = torch.log10(self._stft_magnitude(pred.squeeze(1)).square().clamp(min=1e-6))
        st = torch.log10(self._stft_magnitude(target.squeeze(1)).square().clamp(min=1e-6))

        def _lsd(a, b):
            return (a - b).square().mean(dim=1).sqrt().mean()

        lsd_total = _lsd(sp, st)
        lsd_low = _lsd(sp[..., :bin_idx, :], st[..., :bin_idx, :])
        lsd_high = _lsd(sp[..., bin_idx:, :], st[..., bin_idx:, :])
        return lsd_total, lsd_high, lsd_low

    def _calculate_2f_model_metric(self, z_c, x1_c):
        """Calculate 2f-model (PEAQ) metric for a batch of audio samples."""
        if not PEAQ_AVAILABLE:
            return None
        try:
            def _to_2d(x):
                if x.dim() == 3:
                    x = x.squeeze(1) if x.shape[1] == 1 else torch.mean(x, dim=1)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                return torch.clamp(x, -0.99, 0.99)

            z_peaq = _to_2d(z_c.clone())
            x1_peaq = _to_2d(x1_c.clone())

            peaq_model = PEAQBasic(sampling_rate=48000).to(self.device)
            with torch.no_grad():
                mms_2f = peaq_model.compute_mms_2f(z_peaq, x1_peaq, self.device)
            return float(mms_2f.mean()) if torch.is_tensor(mms_2f) else float(mms_2f)
        except Exception as e:
            print(f"Warning: 2f-model calculation failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Logging helpers
    # ------------------------------------------------------------------ #
    def _build_sample_log(self, z, y, x1, idx, num_steps, prefix="val_samples"):
        """Create wandb log payload for a single audio sample."""
        return {
            f"{prefix}/{idx}/{num_steps}/audio_ground_truth": wandb.Audio(t2n(z), sample_rate=48000),
            f"{prefix}/{idx}/{num_steps}/audio_conditional":  wandb.Audio(t2n(y), sample_rate=48000),
            f"{prefix}/{idx}/{num_steps}/audio_generated":    wandb.Audio(t2n(x1), sample_rate=48000),
            f"{prefix}/{idx}/{num_steps}/spec_ground_truth":  wandb.Image(draw_spec(t2n(z), sr=48000, return_fig=True)),
            f"{prefix}/{idx}/{num_steps}/spec_conditional":   wandb.Image(draw_spec(t2n(y), sr=48000, return_fig=True)),
            f"{prefix}/{idx}/{num_steps}/spec_generated":     wandb.Image(draw_spec(t2n(x1), sr=48000, return_fig=True)),
        }

    # ------------------------------------------------------------------ #
    #  Train step
    # ------------------------------------------------------------------ #
    def _train_step(self, batch_data, step, **kwargs):
        sr_values = batch_data['low_sr']
        lr_bin_count, hf_start_bin = self._get_freq_bins(sr_values[0])

        z = batch_data['hr'].to(self.device)
        y = batch_data['lr_wave'].to(self.device)
        batch_size = z.shape[0]

        Z = self._preprocess(z)
        Y = self._preprocess(y)
        Y_lr, Y_hr, Z_hr = self._split_spectrum(Y, Z, lr_bin_count, hf_start_bin)

        t = torch.rand([batch_size, 1, 1, 1], device=self.device)
        x0 = self.path.sample_source(Y_hr)
        xt = self.path.sample_xt(x0, Z_hr, t)

        output = self.model(xt, t, Y_lr, sr_values)
        target = self.path.get_target_vector_field(xt, x0, Z_hr, t)
        loss = flow_matching_loss(predicted_vf=output, target_vf=target)
        return loss, {"cfm_loss": loss}

    # ------------------------------------------------------------------ #
    #  Validation step
    # ------------------------------------------------------------------ #
    def _val_step(self, batch_data, idx, val_idx, **kwargs):
        ode_steps = kwargs.get('ode_steps', 4)
        val_max_sec = kwargs.get('val_max_sec', 5)

        sr_values = batch_data['low_sr']
        current_sr = sr_values[0]
        lr_bin_count, hf_start_bin = self._get_freq_bins(current_sr)

        z = batch_data['hr'].to(self.device)[..., :48000 * val_max_sec]
        y = batch_data['lr_wave'].to(self.device)[..., :48000 * val_max_sec]
        batch_size = z.shape[0]

        Z = self._preprocess(z)
        Y = self._preprocess(y)
        Y_lr, Y_hr, Z_hr = self._split_spectrum(Y, Z, lr_bin_count, hf_start_bin)

        # CFM loss on random t
        t = torch.rand([batch_size, 1, 1, 1], device=self.device)
        x0 = self.path.sample_source(Y_hr)
        xt = self.path.sample_xt(x0, Z_hr, t)
        output = self.model(xt, t, Y_lr, sr_values)
        target = self.path.get_target_vector_field(xt, x0, Z_hr, t)
        loss = flow_matching_loss(predicted_vf=output, target_vf=target)

        # LSD-high via ODE sampling (same metric used in evaluation)
        lsd_high = None
        with torch.no_grad():
            x1_wave = self._synthesize_waveform(
                Y_lr, Y_hr, sr_values, lr_bin_count, hf_start_bin, ode_steps)
            _, lsd_high, _ = self._compute_lsd(
                x1_wave, z[..., :x1_wave.shape[-1]], current_sr)
            lsd_high = float(lsd_high)

        # Sample logging for selected indices
        log_payload = {}
        if idx in val_idx:
            with torch.no_grad():
                x1_wave_cfg = self._synthesize_waveform(
                    Y[0:1, :, :lr_bin_count, :],
                    Y[0:1, :, hf_start_bin:, :],
                    sr_values, lr_bin_count, hf_start_bin,
                    ode_steps, guidance_scale=1.5)

                min_len = min(z.shape[-1], x1_wave_cfg.shape[-1])
                log_payload = self._build_sample_log(
                    z[0:1, ..., :min_len], y[0:1, ..., :min_len], x1_wave_cfg,
                    idx, ode_steps, prefix="val_samples")

        return {'loss': loss, 'lsd_high': lsd_high, 'log_payload': log_payload}

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, val_idx, ode_steps=4, guidance_scale=None,
                 max_batches=None, global_step=0, **kwargs):

        if hasattr(self, '_eval_output_base'):
            del self._eval_output_base

        self.model.eval()
        total_metrics = {'lsd_total': 0.0, 'lsd_high': 0.0, 'lsd_low': 0.0}
        if PEAQ_AVAILABLE:
            total_metrics['2f_model'] = 0.0
        cnt = 0

        with torch.no_grad():
            eval_pbar = tqdm(self.val_loader, desc='Evaluating...', dynamic_ncols=True)
            for batch_idx, batch in enumerate(eval_pbar):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                outdict = self._evaluate_step(batch, batch_idx, val_idx,
                                              ode_steps=ode_steps,
                                              guidance_scale=guidance_scale)
                lsd_metrics = outdict.get('lsd_metrics', {})
                peaq_metric = outdict.get('2f_model', None)

                if lsd_metrics:
                    for key in total_metrics:
                        if key in lsd_metrics:
                            total_metrics[key] += lsd_metrics[key]
                    cnt += 1

                    postfix = {k: f'{lsd_metrics.get(k, 0):.4f}' for k in ('lsd_total', 'lsd_high', 'lsd_low')}
                    if peaq_metric is not None:
                        postfix['2f_model'] = f'{peaq_metric:.4f}'
                        total_metrics['2f_model'] += peaq_metric
                    eval_pbar.set_postfix(postfix)

                if outdict.get('log_payload'):
                    self.logger.log(outdict['log_payload'], step=global_step)

        avg_metrics = {k: v / cnt if cnt > 0 else 0.0 for k, v in total_metrics.items()}
        self.logger.log({f'eval/{k}': v for k, v in avg_metrics.items()}, step=global_step)

        print(f"Evaluation finished over {cnt} batches.")
        for k, v in avg_metrics.items():
            print(f"   {k}: {v:.4f}")

    def _evaluate_step(self, batch_data, idx, val_idx, **kwargs):
        ode_steps = kwargs.get('ode_steps', 5)
        guidance_scale = kwargs.get('guidance_scale', None)
        val_max_sec = kwargs.get('val_max_sec', 5)

        sr_values = batch_data['low_sr']
        current_sr = sr_values[0]
        lr_bin_count, hf_start_bin = self._get_freq_bins(current_sr)

        z = batch_data['hr'].to(self.device)[..., :48000 * val_max_sec]
        y = batch_data['lr_wave'].to(self.device)[..., :48000 * val_max_sec]

        Z = self._preprocess(z)
        Y = self._preprocess(y)
        Y_lr, Y_hr, _ = self._split_spectrum(Y, Z, lr_bin_count, hf_start_bin)

        # ODE synthesis
        x1_wave = self._synthesize_waveform(
            Y_lr, Y_hr, sr_values, lr_bin_count, hf_start_bin, ode_steps, guidance_scale)

        min_len = min(z.shape[-1], x1_wave.shape[-1])
        z_c = z[..., :min_len]
        x1_c = x1_wave[..., :min_len]

        # LSD metrics (generated vs GT)
        lsd_total, lsd_high, lsd_low = self._compute_lsd(x1_c, z_c, current_sr)
        lsd_metrics = {
            'lsd_total': float(lsd_total),
            'lsd_high': float(lsd_high),
            'lsd_low': float(lsd_low),
        }

        peaq_metric = self._calculate_2f_model_metric(z_c, x1_c) if PEAQ_AVAILABLE else None

        # Save audio
        self._save_audio_samples(z_c, y[..., :min_len], x1_c, batch_data, idx, ode_steps, current_sr)

        # Sample logging
        log_payload = {}
        if idx in val_idx:
            log_payload = self._build_sample_log(
                z_c[0:1], y[..., :min_len][0:1], x1_c[0:1],
                idx, ode_steps, prefix="eval_samples")

        result = {'lsd_metrics': lsd_metrics, 'log_payload': log_payload}
        if peaq_metric is not None:
            result['2f_model'] = peaq_metric
        return result

    def _save_audio_samples(self, z_c, y_c, x1_c, batch_data, batch_idx, ode_steps, current_sr):
        """Save generated audio to disk for offline evaluation."""
        if not hasattr(self, '_eval_output_base'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = getattr(self.logger, 'run_name', 'eval')
            self._eval_output_base = f"samples/outputs/{exp_name}_{current_sr}k_ode{ode_steps}_{timestamp}"
            os.makedirs(self._eval_output_base, exist_ok=True)

        batch_size = z_c.shape[0]
        relpaths = batch_data.get('relpath', [f"batch{batch_idx:03d}_sample{i:02d}.wav" for i in range(batch_size)])
        if not isinstance(relpaths, list):
            relpaths = [relpaths] * batch_size

        for i in range(batch_size):
            out_dir = os.path.join(self._eval_output_base, os.path.dirname(relpaths[i]))
            os.makedirs(out_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(relpaths[i]))[0]
            sf.write(os.path.join(out_dir, f"{base_name}.wav"), t2n(x1_c[i:i+1]).squeeze(), 48000)

        if batch_idx < 3:
            print(f"Saved batch {batch_idx} samples under {self._eval_output_base}")

