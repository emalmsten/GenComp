from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import torch
from gencomp.diffusion.ddpm import DDPM
from PIL import Image
# import imageio

checkpoint = "gencomp\9uvq8fg6\checkpoints\epoch=9-step=74999.ckpt"
path = "configs/mnist.yaml"

config = OmegaConf.load(path)
model = DDPM.load_from_checkpoint(checkpoint_path=checkpoint, config=path)
model.eval()

output = model.sample(batch_size=1, return_intermediates=True, log_every_t= 10)
# print(len(output))
# print(output[0].shape)
# print(len(output[1]))
generated_image_np = output[0].squeeze().cpu().numpy()
plt.imshow(generated_image_np, cmap='gray')
plt.axis('off')
plt.show()


frames = []
for img_array in output[1]:

    # Remove extra dimensions
    img_array = img_array.squeeze().cpu().numpy()
    img_array = (img_array * 255).clip(0, 255)

    img = Image.fromarray(img_array.astype('uint8'))

    frames.append(img)

frames[0].save('output.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)