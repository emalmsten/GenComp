import os
import gzip
import shutil
import torch
from torch.utils.data import ConcatDataset
from lego.lego_dataset import LegoDataset


def load_dataset(processed_data_dir):
    """ Load a processed dataset from the processed_data_dir """
    datasets = []
    dataset_files = os.listdir(processed_data_dir)
    dataset_files.sort()
    for file in dataset_files:
        fp = f"{processed_data_dir}/{file}"

        # Unzip if file is a .gz file
        if file.endswith('.gz'):
            with gzip.open(fp, 'rb') as f:
                dataset = torch.load(f)
        else:
            with open(fp, 'rb') as f:
                dataset = torch.load(f)

        print(f"loaded dataset: {fp}")
        datasets.append(dataset)

    return ConcatDataset(datasets)


def calculate_chunks(dp):
    """ Calculate the chunks to split the dataset into """

    num_samples = dp.num_samples
    num_chunks = dp.num_chunks

    chunk_size = num_samples // num_chunks
    remainder = num_samples % num_chunks  # Remainder to distribute evenly among chunks

    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append((start, end))
        start = end

    return chunks


def unzip_ltron(dir_path, bps):
    """ Unzip the ltron mpd files to the dir_path """
    import tarfile
    import os

    for num_bricks in bps:
        path = f"{dir_path}/ldraw_{num_bricks}"

        # check if files are not already unzipped in dir
        if not os.path.exists(path):
            with tarfile.open(f"{dir_path}/ldraw_{num_bricks}.tar.gz", "r:gz") as tar:
                print(f"unzipping ldraw_{num_bricks} to {path}")
                tar.extractall(path)


def make_dataset(cfg):
    """ Create a dataset from the config file """

    d_cfg = cfg.data.dataset
    ltron_params = cfg.ltron_params
    bricks_per_sample = ltron_params.bricks_per_sample

    ds_dir = d_cfg.params.processed_data_dir
    if d_cfg.save_dataset:
        # Clean out the directory before saving new datset files
        shutil.rmtree(ds_dir, ignore_errors=True)
        os.makedirs(ds_dir, exist_ok=True)

    # only unzip if the files are not already unzipped
    unzip_ltron(d_cfg.ltron_data_path, bricks_per_sample)

    datasets = []
    for i, num_bricks in enumerate(bricks_per_sample):
        chunks = calculate_chunks(d_cfg)
        for start, end in chunks:
            # Create a dataset for each chunk
            dataset = LegoDataset(cfg, num_bricks, start, end)
            datasets.append(dataset)

            fp = f'{ds_dir}/ds_{num_bricks}_{start}_{end}.pth'

            if d_cfg.save_dataset:
                if d_cfg.save_as_gzip:
                    with gzip.open(fp + '.gz', 'wb') as f:
                        torch.save(dataset, f)
                else:
                    with open(fp, 'wb') as f:
                        torch.save(dataset, f)

                print(f"Saved Dataset: {fp}")

    return ConcatDataset(datasets)


def get_dataset(use_processed_dataset, processed_data_dir, whole_config):
    return load_dataset(processed_data_dir) if use_processed_dataset else make_dataset(whole_config)

