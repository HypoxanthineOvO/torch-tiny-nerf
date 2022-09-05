import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnf
from positional_encoding import pos_enc


class NeRF(nn.Module):
    def __init__(self,L_embed = 6,trunk_depth = 8,trunk_width = 256,input_size = 3,output_size = 4,skips = [4]):
        super(NeRF,self).__init__()
        self.D = trunk_depth
        self.W = trunk_width
        self.isz = input_size
        self.osz = output_size
        self.skp = skips
        
        self.linears = nn.ModuleList(
            [nn.Linear(input_size,trunk_width)] +
            [nn.Linear(trunk_width,trunk_width) if i not in self.skp
            else nn.Linear(trunk_width + input_size,trunk_width) for i in range(trunk_depth - 1)
            ]
        )
        
        self.output_linear = nn.Linear(trunk_width,output_size)
        
    def forward(self,inputs):
        inputs_change = inputs
        for i in range(len(self.linears)):
            inputs_change = self.linears[i](inputs_change)
            inputs_change = tnf.relu(inputs_change)
            if i in self.skp:            
                inputs_change = torch.cat([inputs,inputs_change],-1)

        
        outputs = self.output_linear(inputs_change)
        return outputs

    
def create_model(L_embed):
    input_size = 3 + 3 * 2 * L_embed
    model = NeRF(input_size= input_size)
    #model = model.double()
    grad_vars = list(model.parameters())
    optimizer = torch.optim.Adam(params = grad_vars,lr = 5e-3)
    
    return model,optimizer