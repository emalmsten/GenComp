from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import torch
import imageio
from PIL import Image
from tqdm import tqdm
from crosscut_dataset import CrosscutDataset, base_save_samples
from torch.utils.data import DataLoader
import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd() + "/code")
from gencomp.diffusion.ddpm import DDPM

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize dataset and dataloader
dataset = CrosscutDataset('train', rotation=True, use_image_features=False, data_path='./datasets/puzzlefusion', numSamples=1)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

# Load a sample from the dataloader and move to device
original_sample, conditions = next(iter(dataloader))
original_sample = original_sample.to(device)
# Move each item in conditions to device (if conditions is a dictionary or similar structure)
conditions = {k: v.to(device) for k, v in conditions.items()}

# Load model and move to device
checkpoint = "experiments/puzzlefusion/best_model-v2.ckpt"
path = "code/configs/gencomp/puzzlefusion.yaml"
config = OmegaConf.load(path)
model = DDPM.load_from_checkpoint(checkpoint_path=checkpoint, config=path)
model = model.to(device)  # Move model to device
model.eval()

# Sample with the model, ensuring all inputs are on the same device
output = model.sample(batch_size=1, return_intermediates=True, log_every_t=10, **conditions)

# Prepare samples for saving
original_sample = original_sample.unsqueeze(0).permute(0, 1, 3, 2)  # Adjust dimensions as needed
output_sample = output[0].unsqueeze(0).permute(0, 1, 3, 2)
conditions = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in conditions.items()}
# Save samples
base_save_samples(output_sample.cpu(), 'sample', conditions, True, 0, save_edges=True)
base_save_samples(original_sample.cpu(), 'original', conditions, True, 0, save_edges=True)

# Generate frames for the GIF
frames = []
for i, img_array in enumerate(tqdm(output[1])):
    img_array = img_array.unsqueeze(0).permute(0, 1, 3, 2).to(device)  # Ensure each frame is on device
    img = base_save_samples(img_array.cpu(), 'gif', conditions, True, i, save_edges=True, output_path='./code/puzzlefusion/outputs')
    frames.append(img)

# Save the GIF
frames[0].save('code/puzzlefusion/outputs/sample.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
