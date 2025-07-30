import os
import shutil
import re
import argparse
import sys

import torch
from omegaconf import OmegaConf

# change dir
from torch.utils.data import DataLoader

def visualize_dir(dir_path, cfg):
    print(f"Visualizing dir: {dir_path}")

    def natural_key(filename):
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

    play_files = sorted(os.listdir(dir_path), key=natural_key)
    play_files = [f"{dir_path}/{file}" for file in play_files]
    visualize_output(play_files[0], play_files, cfg)


def get_file(cfg):
    from lego.lego_util import get_mpd_path

    v_cfg = cfg.visualize_params
    if v_cfg.file_to_visualize is not None:
        return v_cfg.file_to_visualize

    c_cfg = v_cfg.construct_params
    if c_cfg.random_constructions:
        return get_mpd_path(c_cfg.random_params.idx, v_cfg.num_bricks, cfg)

    # make omr
    omr_cfg = c_cfg.omr_params
    return f"{omr_cfg.path}/{omr_cfg.name}__{omr_cfg.model}_{v_cfg.num_bricks}_{omr_cfg.num}.mpd"


def visualize_output(og_file_path, file_paths, cfg):
    import ltron.visualization.ltron_viewer as ltron_viewer
    from ltron.bricks.brick_scene import BrickScene

    scene = BrickScene(renderable=False, collision_checker=False, track_snaps=True)
    scene.instances.clear()
    scene.import_ldraw(og_file_path)

    ltron_viewer.start_viewer(
        og_file_path,
        1024,
        1024,
        'grey_cube',
        (256, 256, 256),
        1024,
        play_files=file_paths,
        gif_fps=cfg.visualize_params.gif_fps,
        save_dir=cfg.visualize_params.save_dir
    )


def create_intermediates_from_sample(sample_path, sample, conditionals, model, cfg):
    from lego.lego_util import intermediates_to_mpd
    from ltron.bricks.brick_scene import BrickScene

    v_cfg = cfg.visualize_params
    num_corners = cfg.ltron_params.bbox_corners

    scene = BrickScene(
        renderable=False,
        collision_checker=False,
        track_snaps=True)
    scene.clear_instances()
    scene.import_ldraw(sample_path)

    noised_paths, denoised_paths = [], []

    # TODO, could get rid of some duplicate code here
    if v_cfg.show_noised:
        noised = model.visualize_noise(sample, steps=1, config=cfg)
        noised = torch.stack(noised).squeeze(1)[:, :, :v_cfg.num_bricks * num_corners]

        fp = f"{v_cfg.save_dir}/noised"
        shutil.rmtree(fp, ignore_errors=True)
        os.makedirs(fp, exist_ok=True)

        noised_paths = intermediates_to_mpd(scene, noised, v_cfg.num_bricks, fp, cfg)

    if v_cfg.show_denoised:
        _, denoised = model.sample(x_start=sample, config=cfg, batch_size=1, return_intermediates=True, log_every_t=1,
                                   **conditionals)
        denoised = torch.stack(denoised).squeeze(1)[:, :, :v_cfg.num_bricks * num_corners]

        fp = f"{v_cfg.save_dir}/denoised"
        shutil.rmtree(fp, ignore_errors=True)
        os.makedirs(fp, exist_ok=True)

        denoised_paths = intermediates_to_mpd(scene, denoised, v_cfg.num_bricks, fp, cfg)

    return noised_paths, denoised_paths


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


def main(config_dir="./code/configs/lego.yaml", root_dir=None):
    setup(config_dir, root_dir)
    from gencomp.diffusion.ddpm import DDPM
    from lego.lego_util import get_latest_checkpoint
    from lego.lego_dataset import LegoDataset

    if root_dir is not None:
        os.chdir(root_dir)

    cfg = OmegaConf.load(config_dir)
    v_cfg = cfg.visualize_params

    if v_cfg.dir_to_visualize is not None:
        visualize_dir(v_cfg.dir_to_visualize, cfg)
        return

    sample_path = get_file(cfg)

    # load data
    dataset = LegoDataset(cfg, v_cfg.num_bricks, manual_file_paths=[sample_path])
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    sample, conditionals = dataloader.__iter__().__next__()
    sample = sample.to(cfg.trainer.accelerator)
    conditionals = {key: value.to(cfg.trainer.accelerator) for key, value in conditionals.items()}

    checkpoint = v_cfg.path_to_model
    if v_cfg.path_to_model is None:
        checkpoint = get_latest_checkpoint(cfg)

    # load model
    model = DDPM.load_from_checkpoint(checkpoint_path=checkpoint, config=config_dir)
    model.to(cfg.trainer.accelerator)
    model.eval()

    file_paths = [sample_path]
    noised_paths, denoised_paths = create_intermediates_from_sample(sample_path, sample, conditionals, model, cfg)
    file_paths.extend(noised_paths)
    file_paths.extend(denoised_paths)

    visualize_output(sample_path, file_paths, cfg)


if __name__ == "__main__":
    main(config_dir="./code/configs/lego.yaml")