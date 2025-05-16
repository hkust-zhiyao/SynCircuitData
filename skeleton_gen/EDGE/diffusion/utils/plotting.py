import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def get_image_grid(images, nrow=8, padding=2):
    
    image_grid = vutils.make_grid(images, nrow=nrow, padding=padding)
    image_grid = image_grid.permute([1,2,0]).detach().cpu().numpy()
    return image_grid



def plot_quantized_images(images, num_bits=8, nrow=8, padding=2):
    
    image_grid = get_image_grid(images.float()/(2**num_bits - 1), nrow=nrow, padding=padding)
    plt.figure()
    plt.imshow(image_grid)
    plt.show()
