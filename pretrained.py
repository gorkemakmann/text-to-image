# Library imports

# Importing PyTorch library, for building and training neural networks
import torch

# Importing StableDiffusionPipeline to use pre-trained Stable Diffusion models
from diffusers import StableDiffusionPipeline

# Image is a class for the PIL module to visualize images in a Python Notebook
from PIL import Image

pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                  torch_dtype=torch.float16)


# Defining function for the creation of a grid of images
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    
    w, h = imgs[0].size
    grid = Image.new('RGB', size = (cols*w,
                                   rows * w))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box = (i%cols*w, i // cols*h))
    return grid


n_images = 6 # Let's generate 6 images based on the prompt below
prompt = ['Sunset on a beach'] * n_images

images = pipeline(prompt).images

grid = image_grid(images, rows=2, cols = 3)
grid