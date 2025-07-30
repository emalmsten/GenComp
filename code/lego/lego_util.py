from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import torch

# This is the naming in LTRON
file_prepends = {
    2: "pico",
    4: "nano",
    8: "micro",
}


def get_mpd_path(idx, num_bricks, cfg):
    # Based on LTRON naming
    return f"{cfg.data.dataset.ltron_data_path}/ldraw_{num_bricks}/" + file_prepends[num_bricks] + f"_{str(idx).zfill(6)}.mpd"

def get_translations(transforms):
    """Extracts the translations from the transformation matrices"""
    return transforms[..., :3, 3]

def get_rotations(Ts):
    """Extracts the Euler angles from the transformation matrices"""

    # Input needs to be of shape (_, 4, 4) for as_euler to work
    original_shape = Ts.shape[:2]
    reshaped_matrices = Ts.reshape(-1, 4, 4)

    # Extract the 3x3 rotation matrices
    rotation_matrices = reshaped_matrices[:, :3, :3]

    # Convert rotation matrices to Euler angles using SciPy's Rotation
    # TODO, could could be written in PyTorch to optimize
    rotation = R.from_matrix(rotation_matrices)
    euler_angles = rotation.as_euler('xyz')

    # Reshape the output back to orginal, except (_, 3) instad of (_, 4, 4) as last dimensions
    euler_angles = euler_angles.reshape(*original_shape, 3)
    return euler_angles

def normalize(data, normalizing_factor):
    return data / normalizing_factor


def intermediates_to_mpd(scene, intermediates, num_bricks, fp, cfg):
    """Saves the intermediates of one sample to file"""
    file_paths = []
    intermediates = intermediates.permute(0, 2, 1)

    # Necessary for natural sorting later
    zfill_len = np.log10(len(intermediates)).astype(int) + 1

    for i, sample in enumerate(intermediates):
        # Retrives the transformation matrices for the bricks given a sample
        Ts = get_new_transformation_matrices(sample, num_bricks, cfg.ltron_params)

        # Sets the transformation matrices to the original bricks of the scene
        set_new_transformation_matrices(scene, Ts)

        # Create the MPD
        file = f"{fp}/{str(i).zfill(zfill_len)}.mpd"
        file_paths.append(file)
        scene.export_ldraw(file)

    return file_paths

def get_new_transformation_matrices(sampled_bricks, num_bricks, ltron_params):
    """Returns the transformation matrices for the translations and rotations in the sample"""
    # Identity matrices
    Ts = np.array([np.eye(4) for _ in range(num_bricks)])

    for i in range(num_bricks):
        # Skip in sizes of bbox corners since all bricks between are the same
        sampled_brick = sampled_bricks[i * ltron_params.bbox_corners].cpu().numpy()
        # Normalize back to original size
        angles = sampled_brick[3:] * np.pi

        # Create the 4x4 homogeneous transformation matrix
        Ts[i][:3, :3] = R.from_euler('xyz', angles).as_matrix()
        Ts[i][:3, 3] = np.array(sampled_brick[0:3]) * ltron_params.normalization_factor

    return Ts


def set_new_transformation_matrices(scene, Ts):
    """Sets the transformation matrices to the bricks in the scene"""
    all_bricks = scene.instances.instances

    for i, instance in enumerate(all_bricks.values()):
        all_bricks.get(i + 1).transform = Ts[i]
        scene.update_instance_snaps(instance)

def get_latest_checkpoint(cfg):
    """Get the path to the latest checkpoint file in the given directory"""

    # Search for all checkpoint files in the given directory
    ckpt_dir = f"{cfg.wandb.params.save_dir}/ckpts"
    assert os.path.exists(ckpt_dir), f"No checkpoint directory at {ckpt_dir}, have you trained a model yet?"
    checkpoint_files = [f"{ckpt_dir}/{f}" for f in os.listdir(ckpt_dir)]

    # Return the latest checkpoint file based on creation time
    checkpoint_file = max(checkpoint_files, key=os.path.getctime)
    print(f"Loading checkpoint: {checkpoint_file}")
    return checkpoint_file
