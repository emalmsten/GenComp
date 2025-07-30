import networkx as nx
import os
import shutil

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from evaluation import run_evaluation
from gencomp.diffusion.ddpm import DDPM
import time

from lego.graph_creation import create_graphs
from lego.lego_util import intermediates_to_mpd, get_mpd_path, get_latest_checkpoint
from ltron.bricks.brick_scene import BrickScene

def mse(image1, image2):
    return torch.mean((image1 - image2) ** 2)

def remove_temp_intermediates(cfg):
    """Remove all files in the temp_intermediates folder"""
    temp_folder = f"{cfg.testing.output_dir}/temp_intermediates"
    shutil.rmtree(temp_folder, ignore_errors=True)

def save_graphs(graphs, cfg, num_bricks, org_file_idx):
    """Save the graphs to a file"""
    fp = f"{cfg.testing.output_dir}/graphs/{num_bricks}_bricks/sample_{str(org_file_idx).zfill(6)}/"
    os.makedirs(fp, exist_ok=True)
    zfill_len = np.log10(len(graphs)).astype(int) + 1
    for i, G in enumerate(graphs):
        nx.write_graphml(G, f"{fp}/{str(i).zfill(zfill_len)}.graphml")

def save_metrics(res_dicts, cfg):
    """Save the metrics to a file"""
    file = f"{cfg.testing.output_dir}/test_metrics.csv"
    pd.DataFrame(res_dicts).to_csv(file, index=False)
    print(f"metrics saved to {file}")

def get_file_dir_intermediates(cfg, num_bricks, org_file_idx):
    """Get the file path for the intermediates"""
    int_dir = "intermediates" if cfg.testing.save_intermediates else "temp_intermediates"
    fp = f"{cfg.testing.output_dir}/{int_dir}/{num_bricks}_bricks/sample_{str(org_file_idx).zfill(6)}"
    os.makedirs(fp, exist_ok=True)
    return fp


def test_model(path_to_model, num_samples, test_set, whole_config, config_path):
    cfg = whole_config
    t_cfg = cfg.testing
    l_cfg = cfg.ltron_params

    device = torch.device(cfg.trainer.accelerator)
    test_loader = DataLoader(test_set, **t_cfg.loader_params)
    print(f"Testing model on {device}...")

    # If no path to model is given, get the latest checkpoint
    if path_to_model is None:
        path_to_model = get_latest_checkpoint(cfg)
    print(f"Loading model from {path_to_model}")

    # Load the model and the checkpoint
    model = DDPM.load_from_checkpoint(checkpoint_path=path_to_model, config=config_path)
    model.to(device)
    model.eval()

    scene = BrickScene(
        renderable=False,
        collision_checker=False,
        track_snaps=True)

    start_time = time.time()
    res_dicts = []
    with torch.no_grad():
        for i, (datas, conds) in enumerate(test_loader):
            # If not wanting to test on all samples in the test set
            if i == num_samples:
                break

            num_bricks = conds["num_bricks"][0].item()
            org_file_idx = conds["id"][0].item()
            print(f"Testing batch {i+1}/{min(len(test_loader), num_samples)} on {org_file_idx} with {num_bricks} after {time.time() - start_time} seconds")

            # import the original file
            org_mpd_path = get_mpd_path(org_file_idx, num_bricks, cfg)
            scene.clear_instances()
            scene.import_ldraw(org_mpd_path)

            target, conditions = test_loader.__iter__().__next__()
            target = target.to(device)
            conditions = {key: value.to(device) for key, value in conditions.items()}

            output, intermediates = model.sample(batch_size=t_cfg.loader_params.batch_size, return_intermediates=True, log_every_t = 1, **conditions)
            intermediates = torch.stack(intermediates).squeeze(1)[:, :, :num_bricks * l_cfg.bbox_corners]

            # Must first create MPDs before creating graphs, to use LTRON method for checking snap connections
            mpd_files = intermediates_to_mpd(scene, intermediates, num_bricks,
                                             get_file_dir_intermediates(cfg, num_bricks, org_file_idx), cfg)
            graphs = create_graphs(mpd_files, scene)

            merr, mcs, fcr = run_evaluation(graphs)

            output = output[0][:, :num_bricks * l_cfg.bbox_corners]
            target = target[0][:, :num_bricks * l_cfg.bbox_corners]
            mse_score = mse(output, target).item()

            remove_temp_intermediates(cfg)
            if t_cfg.save_graphs:
                save_graphs(graphs, cfg, num_bricks, org_file_idx)

            res_dicts.append({
                "file_idx": org_file_idx,
                "num_bricks": num_bricks,
                "mse": mse_score,
                "merr": merr,
                "mcs": mcs,
                "fcr": fcr,
            })
            print(res_dicts[-1])

            # Always save metrics for backup
            save_metrics(res_dicts, cfg)
