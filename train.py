import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import time
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from load_data import load_tiny_nerf_data
from tqdm import tqdm_notebook as tqdm
from mlp_network import create_model
from rays import get_rays,render_rays

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

images,poses,(H,W),focal,(testimg,testpose) = load_tiny_nerf_data()
model,optimizer = create_model()


N_samples = 64
N_iters = 1000
psnrs = []
iternums = []
i_plot = 25


t = time.time()
for i in range(N_iters+1):
    img_i = np.random.randint(images.shape[0])
    target = images[img_i]
    pose = poses[img_i]
    #print(target.device)
    #print(pose.device)    
    rays_o,rays_d = get_rays(H,W,focal,pose)

    
    rgb, depth, acc = render_rays(model, rays_o, rays_d,zmin= 2.,zmax= 6.,N_samples= N_samples)
    loss = torch.mean(torch.square(rgb - testimg))
    psnr = -10. * torch.log10(loss)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if i%i_plot==0:
        print(i, (time.time() - t) / i_plot, 'secs per iter')
        t = time.time()
        
        # Render the holdout view for logging
        rays_o, rays_d = get_rays(H, W, focal, testpose)
        rgb, depth, acc = render_rays(model, rays_o, rays_d,zmin= 2.,zmax= 6.,N_samples= N_samples)
        loss = torch.mean(torch.square(rgb - testimg))
        
        psnr = -10. * torch.log10(loss)
        plt.imshow(rgb.cpu().detach().numpy())
        plt.title(f'Iteration: {i}')
        plt.show()
        

print('Done')
