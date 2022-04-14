# -*- coding: utf-8 -*-
"""
A more thoroughly, expanded annotated learning version of https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torchvision.transforms as transforms

# Setting Seed: Allow for 'deterministic' random variables for reproducibility
seed = 999
random.seed(seed)
torch.manual_seed(seed)

# Data
class ImageData:
    
    def __init__(self, root, transforms=None, **kwargs):
        """
        Extension for defining and getting image data in ImageFolder object form torch 
        
        Keyword Arguments:
            root (str): Path to directory adhering to torchvision.datasets.ImageFolder specification
            **kwargs: Additional ImageFolder keyword arguments
        """
        
        self.root = root
        
        self.transforms = transforms
        self.data = ImageFolder(self.root, self.transforms, **kwargs)
        
    def get_dataloader(self, **kwargs):
        """ Return Dataloader Object for model evaluation and training """
        
        return torch.utils.data.DataLoader(self.data, **kwargs)
    
    def visualize(self, idx):
        """ DOCSTRING """
        
        if self.transforms and any(isinstance(method, transforms.ToTensor) for method in self.transforms.transforms):
            imgs = torch.zeros(len(idx), *self[idx[0]][0].shape)
            for i, img in enumerate(idx):
                imgs[i, :, :, :] = self[img][0]
        else:
            imgs = torch.zeros(len(idx), 3, *self[idx[0]][0].size) # TODO: Extend to 1 channel grayscale?
            for i, img in enumerate(idx):
                imgs[i, :, :, :] = np.transpose(transforms.ToTensor()(self[img][0]), (1,2,0))
        
        grid = 1
        while grid**2 <= len(imgs): # Create nearest uniform grid space for visualization
            grid += 1
            
        plt.figure(figsize=(grid,grid))
        plt.axis("off")
        
        plt.imshow(np.transpose(make_grid(imgs, padding=2, normalize=True).cpu(),(1,2,0)))
        
        
    def __getitem__(self, idx):
        """ Extend ImageData __getitem__ to ImageData obj """
        return self.data[idx]
    
data = ImageData('img_align_celeba/') # transform=transforms.Compose([transforms.ToTensor()])