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
from universr.utils.utils import count_model_params, print_config

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description="Flow-Matching Model Training Script")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the training configuration file.")
    parser.add_argument('--wandb', type=lambda x: x.lower() == 'true', default=False, help="Set to 'true' to enable WandB logging.")
    return parser.parse_args()

def load_config(config_path: str) -> Box:
    with open(config_path, "r") as file:
        return Box(yaml.safe_load(file))

def main():
    args = parse_args()
    config = load_config(args.config)
    
    torch.manual_seed(config.get('seed', 42))
    random.seed(config.get('seed', 42))
    print_config(config)

    # logger and dataloader
    logger = get_logger(config, args.wandb)
    train_loader, val_loader = prepare_dataloader(config)

    # model, path, and trainer initialization
    transform = AmplitudeCompressedComplexSTFT(**config.transform)
    path = get_path(config.path)
    model = ConvNeXtUNetCond(**config.model)
    print(f"Conditioning encoder params: {count_model_params(model.conditioning_encoder):.2f}M")
    print(f"Model params: {count_model_params(model):.2f}M")
    
    trainer = STFTTrainer(
        path=path,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        transform=transform,
        device=torch.device(DEVICE),
        logger=logger,
    )
    
    trainer.train(
        optimizer_config=config.optimizer,
        scheduler_config=config.scheduler,
        **config.train,
    )
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()