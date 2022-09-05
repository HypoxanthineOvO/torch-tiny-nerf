import numpy as np
import torch

def pos_enc(x,L_embed):
    rets = [x]
    for i in range(L_embed):
        for func in [torch.sin,torch.cos]:
            rets.append(func(2.0 **i *x))
    return torch.cat(rets,dim = -1)