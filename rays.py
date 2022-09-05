import numpy as np
import torch
from positional_encoding import pos_enc

def get_rays(H,W,focal,c2w):
    i,j = torch.meshgrid(torch.arange(0,W,dtype=torch.float32),torch.arange(0,H,dtype=torch.float32),indexing = 'xy')
    if torch.cuda.is_available():
        i = i.to("cuda")
        j = j.to("cuda")
    dirs = torch.stack([(i - W/2)/focal, -(j-H/2)/focal,-torch.ones_like(i)],-1)
    rays_d = torch.sum(dirs[...,np.newaxis,:] * c2w[:3,:3],-1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o,rays_d

def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.

    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod

def render_rays(netfn,rays_o,rays_d,zmin,zmax,N_samples):
    def batchify(func,chunk = 1024*32):
        return lambda inputs:torch.cat([func(inputs[i:i+chunk]) for i in range(0,inputs.shape[0],chunk)],0)
    
    z_values = torch.linspace(zmin,zmax,N_samples)
    if torch.cuda.is_available():
        z_values = z_values.to("cuda")
    pts = rays_o[...,np.newaxis,:] + rays_d[...,np.newaxis,:] * z_values[...,:,np.newaxis]
    
    if torch.cuda.is_available():
        pts = pts.to("cuda")
    
    pts_flat = torch.reshape(pts,[-1,3])
    pts_flat = pos_enc(pts_flat,L_embed= 6)
    raw = batchify(netfn)(pts_flat)
    raw = torch.reshape(raw,list(pts.shape[:-1]) + [4])
    
    sig_a = torch.nn.functional.relu(raw[...,3])
    rgb = torch.sigmoid(raw[...,:3])
    
    one_e_10 = torch.tensor([1e10])
    if torch.cuda.is_available():
        one_e_10 = one_e_10.to("cuda")
    dists = torch.cat([z_values[...,1:] - z_values[...,:-1],one_e_10.expand(z_values[...,:1].shape)],dim = -1)
    alpha = 1.0 - torch.exp(-sig_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)
    #weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    rgb_map = (weights[..., None] * rgb).sum(dim = -2)
    depth_map = (weights * z_values).sum(dim = -1)
    acc_map = weights.sum(-1)
    
    return rgb_map,depth_map,acc_map