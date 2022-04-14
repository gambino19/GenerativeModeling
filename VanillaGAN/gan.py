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
    
    def __init__(self, root, transform=None, **kwargs):
        """
        Extension for defining and getting image data in ImageFolder object form torch 
        
        Notes:
            Transforms is the only kwarg for ImageFolder explicitly defined since
            leaving default as None automatically applies the ToTensor() transfomation
            since will be needed for evaluation
        
        Keyword Arguments:
            root (str): Path to directory adhering to torchvision.datasets.ImageFolder specification
            transform (torchvision.transforms.transorms.Compose): Transformations to apply on data
            **kwargs: Additional ImageFolder keyword arguments
            
        Examples:
            >>> data = ImageData(root='data/',
                                 transform=transforms.Compose([
                                    transforms.Resize(64),
                                    transforms.CenterCrop(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        """
        
        self.root = root
        
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transform
        
        self.data = ImageFolder(self.root, self.transform, **kwargs)
        
    def get_dataloader(self, **kwargs):
        """ Return Dataloader Object for model evaluation and training """
        
        return torch.utils.data.DataLoader(self.data, **kwargs)
    
    def visualize(self, idx):
        """ 
        Visualize ImageData by Index
        
        Keyword Arguments:
            idx (int, list): Single index or list of indices to visualize
        """
        
        imgs = torch.zeros(len(idx), *self[idx[0]][0].shape)
        for i, img in enumerate(idx):
            imgs[i, :, :, :] = self[img][0]
        
        nrow = 1
        while nrow**2 <= len(imgs): # Create nearest uniform grid space for visualization
            nrow += 1
            
        print(nrow)
            
        plt.figure(figsize=(nrow,nrow))
        plt.axis("off")
        
        plt.imshow(np.transpose(make_grid(imgs, nrow=nrow, padding=2, normalize=True).cpu(),(1,2,0)))
        
        
    def __getitem__(self, idx):
        """ Extend ImageFolder __getitem__ to ImageData obj """
        return self.data[idx]
    
    def __len__(self):
        """ Extend ImageFolder __len__ to ImageData obj """
        return len(self.data)
    
data = ImageData('/') # transform=transforms.Compose([transforms.ToTensor()])