import torch as th
from crosscut_dataset import  CrosscutDataset
from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
import argparse
import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd()+"/code")
from gencomp.diffusion.ddpm import DDPM
import torch
from lightning.pytorch.loggers import WandbLogger
import wandb

def main():
    device = "gpu" if torch.cuda.is_available() else "cpu"

    experiment = 'train'
    batch_size = 256
    numSamples = 1
    epochs = 1

    # train_ds = MNIST("data/mnist", train=True, download=True, transform=transforms.ToTensor())
    # train_loader = DataLoader(train_ds, batch_size=8)
    dataset = CrosscutDataset(experiment, rotation=True, use_image_features=False, data_path='./datasets/puzzlefusion', numSamples=numSamples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    path = "./code/configs/gencomp/puzzlefusion.yaml"
    config = OmegaConf.load(path)
    # config = OmegaConf.merge(*configs, None)

    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_opt = argparse.Namespace(**trainer_config)

    model = DDPM(path)
    wandb_logger = WandbLogger(log_model="all", project="gencomp-puzzlefusion", save_dir="experiments/puzzlefusion", checkpoint_name=f"puzzlefision_exp.{experiment}_nSamp.{numSamples}_bs.{batch_size}_ep.{epochs}")

    trainer = Trainer(
        max_epochs=epochs,
        devices=1,       # Number of devices (CPU)
        accelerator=device,  # Specify CPU as the accelerator
        logger = wandb_logger
    )

    trainer.fit(model, dataloader)
    wandb.finish()

if __name__ == '__main__':
    main()