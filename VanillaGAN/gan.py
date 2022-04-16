# -*- coding: utf-8 -*-
"""
A more thoroughly, expanded annotated learning version of https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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
        
        idx = [idx] if isinstance(idx, int) else idx
        
        imgs = torch.zeros(len(idx), *self[idx[0]][0].shape)
        for i, img in enumerate(idx):
            imgs[i, :, :, :] = self[img][0]
        
        nrow = 1
        while nrow**2 <= len(imgs): # Create nearest uniform grid space for visualization
            nrow += 1
            
        plt.figure(figsize=(nrow,nrow))
        plt.axis("off")
        
        plt.imshow(np.transpose(make_grid(imgs, nrow=nrow, padding=2, normalize=True).cpu(),(1,2,0)))
        
    def __getitem__(self, idx):
        """ Extend ImageFolder __getitem__ to ImageData obj """
        return self.data[idx]
    
    def __len__(self):
        """ Extend ImageFolder __len__ to ImageData obj """
        return len(self.data)
    

def weights_init(m, weight_by=nn.init.normal_, mean=0.0, std=0.02, constant=0, **kwargs):
    """ 
    Apply weight initialization across model components
    
    Note:
        Usage of method for initialization depends on whether any arguments are passed
        If no arguments are passed, can use base .apply method in model
        If arguments are passed, have to use a lambda expression for apply
        
    Keyword Arguments:
        m (nn.Module): Initialized model object
        weight_by (nn.init method): Weighting method. Default nn.init.normal_
        mean (float):  Default Keyword argument - mean of the normal distribution 
        std (float): Default Keyword argument - the standard deviation of the normal distribution
        constant (float): Constant for bias
        **kwargs: Additional keyword arguments for weighting method if not default
        
    Example:
        >>> m.apply(weights_init) # If no arguments are needed
        >>> m.apply(lambda m: set_dropout(m, weight_by=nn.init.uniform_, a=0.0, b=1.0)) # If arguments are needed
    """
    
    if weight_by is nn.init.normal_: # Default
        params = {'mean': mean, 'std': std}
    else:
        params = kwargs.copy()
        
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_by(m.weight.data, **params)
    elif classname.find('BatchNorm') != -1:
        weight_by(m.weight.data, **params)
        nn.init_constant_(m.bias.data, constant)


data = ImageData('data/') # transform=transforms.Compose([transforms.ToTensor()])