import sys
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
sys.path.insert(0, os.getcwd()) 
from protein.datasets import ProteinAngleDataset
from gencomp.diffusion.ddpm import DDPM
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
import argparse
os.chdir('..')

def create_repeated_dataset(original_dataset, repeat_times):
    # Get the first sample from the original dataset
    first_sample = original_dataset[0]
    # Create a list that repeats the first sample
    repeated_data = [first_sample] * repeat_times
    return repeated_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 200
    batch_size = 32
    # Original Dataset
    original_dataset = ProteinAngleDataset(data_path='/mnt/c/Users/chenwanxin/Documents/GitHub/GenComp/protein/cath/small')
    # Create the repeated dataset using the first sample
    repeated_dataset = create_repeated_dataset(original_dataset, repeat_times=256)
    
    # Build DataLoader
    dataloader = DataLoader(repeated_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    path = "/mnt/c/Users/chenwanxin/Documents/GitHub/GenComp/configs/gencomp/protein.yaml"
    config = OmegaConf.load(path)
    
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_opt = argparse.Namespace(**trainer_config)

    model = DDPM(config.model)
    wandb_logger = WandbLogger(log_model="all", project="protein_test")

    trainer = Trainer(
        max_epochs=epochs,
        devices=1,       # Number of devices (GPU)
        accelerator="gpu",  # Specify GPU as the accelerator
        logger=wandb_logger
    )

    trainer.fit(model, dataloader)
    wandb.finish()
    
if __name__ == '__main__':
    main()
