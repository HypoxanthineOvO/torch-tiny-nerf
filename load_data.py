import numpy as np
import torch
import os

def load_tiny_nerf_data():
    data = np.load('tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    H, W = images.shape[1],images.shape[2]
    
    images = torch.from_numpy(images)
    poses = torch.from_numpy(poses)
    focal = torch.from_numpy(focal)
    
    testimg, testpose = images[101],poses[101]
    images = images[:100,...,:3]
    poses = poses[:100]
    
    
    return images,poses,(H,W),focal,(testimg,testpose)
