import os
import torch
import numpy as np
import sys
from omegaconf import OmegaConf
from datasets import ProteinAngleDataset
from gencomp.diffusion.ddpm import DDPM
os.chdir("..")

# Set device and load model checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
checkpoint = "protein_test/8mgydifz/checkpoints/epoch=4999-step=3870000.ckpt"
path = "code/configs/gencomp/protein.yaml"

config = OmegaConf.load(path)
model = DDPM.load_from_checkpoint(checkpoint_path=checkpoint, config=config)
model = model.to(device)

def sample_with_intermediates(model, seq_len, save_dir, batch_size=1, device=None, save_intermediates=False):
    """
    Samples protein sequences with intermediate outputs and saves them to a specified directory.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The DDPM model for sampling.
    seq_len : int
        Length of protein sequence to generate.
    save_dir : str
        Directory path where outputs will be saved.
    batch_size : int
        Number of samples per batch.
    device : torch.device
        The device to run sampling on.
    save_intermediates : bool
        Whether to save intermediate outputs.
    
    Returns:
    --------
    np.ndarray
        The generated sequence output truncated to seq_len.
    list
        List of intermediate outputs if save_intermediates is True.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize masks
    self_mask = torch.ones(batch_size, 1, 128, 128, device=device)
    gen_mask = torch.ones(batch_size, 1, 128, 128, device=device)

    # Sampling with intermediate outputs
    with torch.no_grad():
        output, intermediates = model.sample(
            batch_size=batch_size,
            return_intermediates=False,
            log_every_t=100,
            self_mask=self_mask,
            gen_mask=gen_mask
        )

    # Truncate the final output to the required seq_len
    truncated_output = output[:, :, :seq_len].cpu().numpy()  # Convert to a NumPy array

    # Save final output
    final_output_path = os.path.join(save_dir, f"protein_sample_{seq_len}.npy")
    np.save(final_output_path, truncated_output[0])  # Save the first sample in NumPy format
    print(f"Final protein sample of length {seq_len} saved to {final_output_path}")
    
    # Save intermediates if required
    if save_intermediates and intermediates is not None:
        for step_idx, intermediate in enumerate(intermediates):
            intermediate_path = os.path.join(save_dir, f"intermediate_step_{step_idx}.npy")
            np.save(intermediate_path, intermediate[0].cpu().numpy())
            print(f"Intermediate step {step_idx} saved to {intermediate_path}")
    
    return truncated_output, intermediates

# Set the range of sequence lengths to generate and main save directory
#sequence_lengths = range(50, 128)  
sequence_lengths = range(50,129)
main_save_directory = "code/protein/sampled_proteins"  
# Generate samples for each length
for seq_len in sequence_lengths:
    for sample_idx in range(10):  # Generate 10 samples per length
        print(f"Generating protein of length {seq_len}, sample {sample_idx + 1}/20")
        
        # Define specific save path for each sample
        sample_save_dir = os.path.join(main_save_directory, f"length_{seq_len}", f"sample_{sample_idx + 1}")
        os.makedirs(sample_save_dir, exist_ok=True)
        
        # Call sampling function with specified save directory
        final_output, intermediates = sample_with_intermediates(
            model, 
            seq_len=seq_len,
            save_dir=sample_save_dir,  # Save path specified for each sample
            batch_size=1, 
            device=device,
            save_intermediates=True
        )
        
    print(f"Completed generation for length {seq_len}.")
