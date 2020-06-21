"""
Implementation of GDRQ [CVPR 2020]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .quant import *

def reshape_quant(input, nbit, k=2):
    T = k * input.abs().mean()
    input = input.clamp(-T, T)
    
    scale, zero_point = symmetric_linear_quantization_params(nbit, T, restrict_qrange=True)
    output = linear_quantize(input, scale, zero_point)
    return output, scale


class GDRQ_conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                padding=0, dilation=1, groups=1, bias=False, nbit=4, k=2):
        super(GDRQ_conv2d, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.nbit = nbit
        self.k = k
        self.quant_group = self.weight.size(0)
    
    def forward(self, input):
        w_l = self.weight.clone()
        s = []

        for ii in range(self.quant_group):
            wgs = w_l[i*self.group_chunk:(i+1)*self.group_chunk,:,:,:]
            
            T = self.k * wgs.abs().mean()                   
            wgs = wgs.clamp(-T, T)                                                  # clipping

            scale, zero_point = symmetric_linear_quantization_params(self.nbit, T, restrict_qrange=True)
            wgs_q = STEQuantizer_weight(wgs, scale, zero_point, True, False, self.nbit, True)           # quantization
            
            w_l[i*self.group_chunk:(i+1)*self.group_chunk,:,:,:] = wgs_q
            s.append(scale)
        
        output = F.conv2d(input, w_l, self.bias, self.stride, self.padding, self.dilation, self.groups)
        s = torch.Tensor(s)
        return output, s

