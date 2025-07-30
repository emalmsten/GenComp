from datasets import ProteinAngleDataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from lightning.pytorch import Trainer
import argparse
import os, sys
import torch
sys.path.insert(0, os.getcwd()) 
from gencomp.diffusion.ddpm import DDPM
from lightning.pytorch.loggers import WandbLogger
import wandb

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 5000
    batch_size = 32
    # Dataset
    dataset = ProteinAngleDataset(data_path='./protein/cath/train')
    
    # Build DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    path = "./configs/gencomp/protein.yaml"
    config = OmegaConf.load(path)
    # config = OmegaConf.merge(*configs, None)
    
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_opt = argparse.Namespace(**trainer_config)

    model = DDPM(config.model)
    wandb_logger = WandbLogger(log_model="all", project="protein_test")

    trainer = Trainer(
        max_epochs=epochs,
        devices=1,       # Number of devices (GPU)
        accelerator="gpu",  # Specify GPU as the accelerator
        logger = wandb_logger
    )

    trainer.fit(model, dataloader)
    wandb.finish()
    
if __name__ == '__main__':
    main()