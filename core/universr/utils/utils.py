import os
import random

import librosa
import numpy as np
import torch
import yaml
from box import Box
from matplotlib import pyplot as plt


def count_model_params(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1_000_000


def t2n(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().squeeze().numpy()


def read_file_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_config(config_path):
    with open(config_path, "r") as file:
        return Box(yaml.safe_load(file))


def _worker_init_fn(worker_id):
    base_seed = torch.initial_seed()
    seed = (base_seed + worker_id) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_config(config, indent=0):
    for k, v in config.items():
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            print_config(v, indent + 4)
        else:
            print(" " * indent + f"{k}: {v}")


def draw_spec(x,
              figsize=(7, 4), title='', n_fft=2048,
              win_len=1024, hop_len=512, sr=16000, cmap='inferno',
              window='hann',
              vmin=-50, vmax=40, use_colorbar=False,
              ylim=None,
              title_fontsize=10,
              label_fontsize=8,
              return_fig=False,
              save_fig=False, save_path=None):
    fig = plt.figure(figsize=figsize)
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_len, win_length=win_len, window=window)
    stft = 20 * np.log10(np.clip(np.abs(stft), a_min=1e-8, a_max=None))

    plt.imshow(stft,
               aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
               origin='lower', extent=[0, len(x) / sr, 0, sr // 2])

    if use_colorbar:
        plt.colorbar()

    plt.xlabel('Time (s)', fontsize=label_fontsize)
    plt.ylabel('Frequency (Hz)', fontsize=label_fontsize)

    if ylim is None:
        ylim = (0, sr / 2)
    plt.ylim(*ylim)

    plt.title(title, fontsize=title_fontsize)
    plt.tight_layout()

    if save_fig and save_path:
        plt.savefig(f"{save_path}.png")

    if return_fig:
        plt.close()
        return fig
    else:
        plt.show()
        return stft

