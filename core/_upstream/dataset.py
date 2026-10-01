import os
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

from universr.utils.utils import _worker_init_fn, read_file_list


def make_dataset(config, mode:str):
    return Dataset(
        **config.dataset.common,
        **config.dataset[mode],
        mode=mode,
    )

def prepare_dataloader(config):
    train_dataset = make_dataset(config, 'train')
    val_dataset = make_dataset(config, 'val')

    collator = WaveformCollator(
        target_sr=config.dataset.common.sr,
        sampling_rates_probs=config.collator.sampling_rates_probs,
    )

    dl_args = dict(config.dataloader)
    dl_args['worker_init_fn'] = _worker_init_fn
    dl_args['collate_fn'] = collator
    train_loader = DataLoader(train_dataset, shuffle=True, **dl_args)
    
    collator = WaveformCollator(
        target_sr=config.dataset.common.sr,
        sampling_rates_probs=config.collator.validation_probs,
    )
    val_loader_args = dict(config.dataloader)
    val_loader_args['worker_init_fn'] = _worker_init_fn
    val_loader_args['collate_fn'] = collator
    val_loader_args['batch_size'] = 1
    val_loader = DataLoader(val_dataset, shuffle=False, **val_loader_args)

    return train_loader, val_loader

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 file_list: str,
                 num_samples=24000,
                 sr=24000,
                 mode="train"):
        self.num_samples, self.sr, self.mode = num_samples, sr, mode
        
        self.wb_paths = read_file_list(file_list)
        print(len(self.wb_paths), 'samples loaded!')
        # derive common root directory to preserve input folder structure in outputs
        dirnames = [os.path.dirname(p) for p in self.wb_paths]
        self.common_root = os.path.commonpath(dirnames) if len(dirnames) > 0 else ""

    def __len__(self):
        return len(self.wb_paths)

    def _pad(self, wav, N=80):
        pad = (N - wav.shape[-1] % N) % N
        return torch.nn.functional.pad(wav, (0,pad))

    def _ensure(self, wav, L, repeat=True):
        # if short: repeat, else: crop
        if wav.shape[-1] < L and repeat: 
            wav = torch.nn.functional.pad(wav, (0, 4000)) # offset
            reps = (L + wav.shape[-1] - 1) // wav.shape[-1]          # ceil(L / wav.shape[-1])
            wav = wav.repeat(1, reps)[..., :L]   # repeat
        elif wav.shape[-1] < L and not repeat:
            pad = L - wav.shape[-1]
            wav = torch.nn.functional.pad(wav, (0, pad))        
        elif wav.shape[-1] > L:        
            wav = wav[..., :L]
        return wav

    def __getitem__(self, idx):
        wb_path = self.wb_paths[idx]
        y, sr = torchaudio.load(wb_path)
        if y.size(0) > 1:
            y = y.mean(dim=0, keepdim=True)
    
        # gain & normalize (peak normalize to target dBFS)
        gain = np.random.uniform(-1, -6) if self.mode == 'train' else -3
        peak = y.abs().max().clamp(min=1e-8)
        target_peak = 10 ** (gain / 20.0)
        y = y * (target_peak / peak)
        
        # resample
        if sr != self.sr:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sr)
        
        if self.mode=="train":
            target_signal_len = self.num_samples
            current_signal_len = y.shape[-1]
            if current_signal_len <= target_signal_len:
                y = self._ensure(y, target_signal_len)
            else:
                s = np.random.randint(0, current_signal_len - target_signal_len)
                y = y[..., s:s+target_signal_len]
        elif self.mode in ['val']:
            y = y[...,:48000*5]
        else:
            raise ValueError(f"Unsupported mode '{self.mode}'. Expected 'train' or 'val'.")
         
        outdict = {
            'hr': y,
            'filename': Path(wb_path).stem,
            'path': wb_path,
            'relpath': os.path.relpath(wb_path, self.common_root) if self.common_root else Path(wb_path).name,
        }

        return outdict

class WaveformCollator:
    def __init__(self, 
                 target_sr=48000, 
                 sampling_rates_probs={8: 0.7, 12: 0.1, 16: 0.1, 24: 0.1}):
        """
        Initializes the collator.
        Args:
            target_sr (int): The high-resolution sample rate (e.g., 48000).
            sampling_rates_probs (dict): A dictionary mapping sample rates (in kHz) to their sampling probabilities.
                                         Example: {8: 0.7, 12: 0.1, 16: 0.1, 24: 0.1}
        """
        self.target_sr = target_sr
        self.sampling_rates = list(sampling_rates_probs.keys())  # [8, 12, 16, 24]
        self.probs = list(sampling_rates_probs.values()) # [0.7, 0.1, 0.1, 0.1]

    def _apply_lpf(self, hr_wave, low_sr_khz):
        """
        Applies a low-pass filter by downsampling and then upsampling the waveform.
        This correctly simulates the anti-aliasing filter effect.
        """
        original_len = hr_wave.shape[-1]
        target_sr_hz = low_sr_khz * 1000
        
        # Downsample to the target low sample rate
        lr_wave_resampled = torchaudio.functional.resample(
            hr_wave, orig_freq=self.target_sr, new_freq=target_sr_hz
        )
        # Upsample back to the original high sample rate to match lengths
        lr_wave_upsampled = torchaudio.functional.resample(
            lr_wave_resampled, orig_freq=target_sr_hz, new_freq=self.target_sr
        )
        
        lr_wave_upsampled = lr_wave_upsampled[..., :original_len]
        return lr_wave_upsampled

    def __call__(self, batch):
        """
        Processes a batch of data items from the Dataset.
        """
        # 1. Choose one low_sr for the entire batch based on given probabilities
        low_sr_khz = random.choices(self.sampling_rates, self.probs, k=1)[0]
        
        # 2. Stack HR waveforms from the batch items
        # Assuming dataset returns {'hr': tensor_shape_[1, T], ...}
        hr_waves = torch.stack([item['hr'].squeeze(0) for item in batch])

        # 3. Create LR versions by applying the LPF
        lr_waves = self._apply_lpf(hr_waves, low_sr_khz)
        
        # 4. Return a dictionary that matches the trainer's expectation
        return {
            'hr': hr_waves.unsqueeze(1),       # Shape: [B, 1, T]
            'lr_wave': lr_waves.unsqueeze(1), # Shape: [B, 1, T]
            'low_sr': [low_sr_khz] * len(batch), # [B]
            'filename': [item['filename'] for item in batch], # [B]
            'relpath': [item.get('relpath', item['filename']) for item in batch], # [B]
            'path': [item.get('path', item['filename']) for item in batch], # [B]
        }