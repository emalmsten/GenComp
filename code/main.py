import os
import sys

from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from gencomp.diffusion.ddpm import DDPM
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from gencomp.util import instantiate_from_config
import wandb
import argparse
import torch


def train(train_set, config_path, config):
    train_loader = DataLoader(train_set, **config.data.params)

    model = DDPM(config_path)

    api_key = config.wandb.api_key

    # has to read it from a file not pushed to github
    if hasattr(config.wandb, 'api_key_from_file') and config.wandb.api_key_from_file is not None:
        api_key = open(config.wandb.api_key_from_file).read().strip()

    # if the save dir does not exist, create it
    os.makedirs(config.wandb.params.save_dir, exist_ok=True)

    # save checkpoints here
    dirpath = f"{config.wandb.params.save_dir}/ckpts"
    os.makedirs(dirpath, exist_ok=True)

    wandb.login(key=api_key)
    wandb_logger = WandbLogger(log_model="all", **config.wandb.params)

    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=config.wandb.params.checkpoint_name + '_{epoch:02d}',
        save_top_k=config.wandb.save_top_k,
        mode="min"
    )

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        **config.trainer
    )

    trainer.fit(model, train_loader)
    wandb.finish()

    return model


def setup(config_dir, root_dir):
    parser = argparse.ArgumentParser(description="Process config file.")
    parser.add_argument(
        '-c', '--config',
        type=str,
        default=config_dir,
        help='Path to the configuration file (YAML format)'
    )
    parser.add_argument(
        '-root_dir',
        type=str,
        default=root_dir,
        help="""Path to the root directory of the project, i.e path to GenComp.
                    Only needed if the code is not run from the root directory."""
    )
    args = parser.parse_args()
    if root_dir is not None:
        os.chdir(args.root_dir)

    print(f"running from {os.getcwd()}")

    assert args.config is not None, "Config file not provided."
    assert os.path.exists(args.config), f"No config file at {args.config}. Your cwd is: {os.getcwd()}"

    sys.path.insert(0, os.getcwd())
    sys.path.insert(1, os.getcwd() + "/code")
    sys.path.insert(2, os.getcwd() + "/code/ltron")
    sys.path.insert(3, os.getcwd() + "/code/lego")
    return args


def main(config_dir = None, root_dir = None):
    args = setup(config_dir, root_dir)
    config = OmegaConf.load(args.config)
    if config.general.dataset == 'ltron':
        instantiate_from_config(config.setup)

    data_params = config.data.dataset
    dataset = instantiate_from_config(data_params, whole_config=config)

    torch.manual_seed(data_params.seed)
    train_size = int(data_params.train_size * len(dataset))
    test_size = int(data_params.test_size * len(dataset))

    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    if config.general.train:
        train(train_set, args.config, config)
    if config.general.test:
        instantiate_from_config(config.testing, test_set=test_set, whole_config=config, config_path=args.config)


if __name__ == "__main__":
    main()