import os,sys
#sys.path.insert(0, os.path.abspath('protein'))
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from protein.angles_and_coordinates import coord_to_angles
import random

# Function to normalize angles to [-1, 1]
def normalize_angles(data):
    # Assuming data is in range [-π, π]
    return data / np.pi  # Normalizing to [-1, 1]

# Helper function to load PDB files from a folder
def load_pdb_files(pdb_folder):
    """Load PDB files from the specified folder."""
    pdb_files = glob.glob(os.path.join(pdb_folder, "*.pdb"))
    if not pdb_files:
        print(f"No PDB files found in {pdb_folder}")
    else:
        print(f"Found {len(pdb_files)} PDB files in {pdb_folder}")
    return pdb_files

class ProteinAngleDataset(Dataset):
    def __init__(self, data_path='./datasets/cath/train', transform=None, max_length=128,**kwargs):
        # Load PDB files from the given folder
        self.file_paths = load_pdb_files(data_path)
        self.angles = ["phi", "psi", "omega", "theta_1", "theta_2", "theta_3"]
        self.transform = transform
        self.data = []
        self.success_count = 0  
        self.max_length = max_length  # Set maximum sequence length

        # Process each file and extract angle data
        for file_path in self.file_paths:
            df = coord_to_angles(file_path)
            if df is not None:
                normalized_data = normalize_angles(df.values)  # Normalize angles to [-1, 1]
                self.data.append(torch.tensor(normalized_data, dtype=torch.float32))
                self.success_count += 1
        
        # Print the number of successfully processed files
        print(f"Successfully processed {self.success_count} out of {len(self.file_paths)} files.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        seq_length = sample.size(0)

        # If the sequence is longer than max_length, crop it
        if seq_length > self.max_length:
            start_idx = random.randint(0, seq_length - self.max_length)
            sample = sample[start_idx:start_idx + self.max_length]
        else:
            # If the sequence is shorter than max_length, pad it with zeros
            padding_size = self.max_length - seq_length
            padding_tensor = torch.zeros((padding_size, sample.size(1)), dtype=torch.float32)
            sample = torch.cat([sample, padding_tensor], dim=0)

        # Perform permutation to change the shape from [128, 6] to [6, 128]
        sample = sample.permute(1, 0)  # [6, 128]

        # Create gen_mask and self_mask
        gen_mask = torch.zeros((self.max_length, self.max_length), dtype=torch.float32)
        self_mask = torch.ones((self.max_length, self.max_length), dtype=torch.float32)

        valid_len = seq_length if seq_length < self.max_length else self.max_length  # Get valid length

        if valid_len == 0:
            print(f"Warning: Sequence {idx} has valid_len = 0, gen_mask will be all zeros.")
        
        # Set the valid portion of gen_mask to 1
        gen_mask[:valid_len, :valid_len] = 1  # Mark valid positions
        
        # Set the diagonal of self_mask to 0 for valid positions (allow self-attention)
        for j in range(valid_len):
            self_mask[j, j] = 0  # Allow each token to attend to itself

        cond = {
            'self_mask': self_mask,
            'gen_mask': gen_mask,
        }

        return sample, cond
