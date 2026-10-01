import argparse
import random

import torch
import wandb
import yaml
from box import Box

from data.dataset import prepare_dataloader
from universr.flow.path import get_path
from universr.models.unet import ConvNeXtUNetCond
from universr.trainer.trainer import STFTTrainer
from universr.utils.logger import get_logger
from universr.utils.spectral_ops import AmplitudeCompressedComplexSTFT
from universr.utils.utils import print_config

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description="Flow-Matching Model Evaluation Script")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the evaluation configuration file.")
    parser.add_argument('--wandb', type=lambda x: x.lower() == 'true', default=False, help="Enable WandB logging.")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument('--exp-name', type=str, default=None, help="Experiment name for WandB (overrides config).")
    parser.add_argument('--ode-steps', type=int, default=None, help="Number of ODE steps (overrides config).")
    parser.add_argument('--guidance-scale', type=float, default=None, help="Guidance scale for CFG (overrides config).")
    parser.add_argument('--max-batches', type=int, default=None, help="Max batches to evaluate (overrides config).")
    parser.add_argument('--sampling-rate', type=int, default=None, choices=[8, 12, 16, 24],
                        help="Sampling rate in kHz (overrides config).")
    parser.add_argument('--sample-indices', type=int, nargs='+', default=None,
                        help="Specific batch indices to log (overrides num_log_samples).")
    return parser.parse_args()


def load_config(config_path: str) -> Box:
    with open(config_path, "r") as file:
        return Box(yaml.safe_load(file))


def apply_cli_overrides(config, args):
    if args.exp_name is not None:
        config.wandb.run_name = args.exp_name
    if args.sampling_rate is not None:
        config.collator.validation_probs = {args.sampling_rate: 1.0}

def get_eval_params(config, args, total_batches):
    """Resolve eval parameters from config + CLI overrides."""
    eval_cfg = config.get('eval', {})

    # Sample indices: CLI > config explicit list > auto-generate from num_log_samples
    if args.sample_indices is not None:
        val_idx = set(args.sample_indices)
    else:
        n = eval_cfg.get('num_log_samples', 6)
        val_idx = set(torch.linspace(0, total_batches - 1, n).long().tolist())

    return {
        'ode_steps': args.ode_steps or eval_cfg.get('ode_steps', 4),
        'guidance_scale': (args.guidance_scale if args.guidance_scale is not None
                           else eval_cfg.get('guidance_scale', 1.5)),
        'max_batches': args.max_batches or eval_cfg.get('max_batches', None),
        'val_idx': val_idx,
    }

def info_model(config, model):
    """Print model architecture summary."""
    from torchinfo import summary

    hr_freq_bins = config.model.get('hr_freq_bins', 432)
    sr_to_lr_bins = config.model.get('sr_to_lr_bins', {24: 256})
    max_sr = max(sr_to_lr_bins.keys())
    T = 256

    x = torch.randn(1, 2, hr_freq_bins, T)
    t = torch.randint(0, 1000, (1,))
    y = torch.randn(1, 2, sr_to_lr_bins[max_sr], T)

    print(summary(
        model,
        input_data=[x, t, y, [max_sr]],
        depth=3,
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        verbose=0,
    ))


def main():
    args = parse_args()
    config = load_config(args.config)

    torch.manual_seed(config.get('seed', 42))
    random.seed(config.get('seed', 42))

    apply_cli_overrides(config, args)
    print_config(config)

    logger = get_logger(config, args.wandb)
    train_loader, val_loader = prepare_dataloader(config)
    eval_params = get_eval_params(config, args, len(val_loader))

    # Model setup
    transform = AmplitudeCompressedComplexSTFT(**config.transform)
    path = get_path(config.path)
    model = ConvNeXtUNetCond(**config.model)
    info_model(config, model)
    model = STFTTrainer.load_model_for_inference(model, args.ckpt, device=DEVICE)

    trainer = STFTTrainer(
        path=path,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        transform=transform,
        device=torch.device(DEVICE),
        logger=logger,
    )

    trainer.evaluate(global_step=0, **eval_params)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()