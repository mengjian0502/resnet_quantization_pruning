"""
quant_layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .quantizer import *

__all__ = ['ClippedReLU', 'clamp_conv2d', 'sawb_tern_Conv2d', 'int_conv2d']


class sawb_w2_Func(torch.autograd.Function):

    def __init__(self, alpha):
        super(sawb_w2_Func, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        self.save_for_backward(input)
        
        output = input.clone()
        output[input.ge(self.alpha - self.alpha/3)] = self.alpha
        output[input.lt(-self.alpha + self.alpha/3)] = -self.alpha
        
        output[input.lt(self.alpha - self.alpha/3)*input.ge(0)] = self.alpha/3
        output[input.ge(-self.alpha + self.alpha/3)*input.lt(0)] = -self.alpha/3

        return output
    
    def backward(self, grad_output):
    
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0

        return grad_input

class sawb_ternFunc(torch.autograd.Function):
    def __init__(self, th):
        super(sawb_ternFunc,self).__init__()
        self.th = th

    def forward(self, input):
        self.save_for_backward(input)

        # self.th = self.tFactor*max_w #threshold
        output = input.clone().zero_()
        output[input.ge(self.th - self.th/2)] = self.th
        output[input.lt(-self.th + self.th/2)] = -self.th

        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class ClippedReLU(nn.Module):
    def __init__(self, num_bits, alpha=8.0, inplace=False, dequantize=True):
        super(ClippedReLU, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]))     
        self.num_bits = num_bits
        self.inplace = inplace
        self.dequantize = dequantize
        
    def forward(self, input):
        input = F.relu(input)
        input = torch.where(input < self.alpha, input, self.alpha)
        
        with torch.no_grad():
            scale, zero_point = quantizer(self.num_bits, 0, self.alpha)
        input = STEQuantizer.apply(input, scale, zero_point, self.dequantize, self.inplace)
        return input
    
class clamp_conv2d(nn.Conv2d):

    def forward(self, input):
        
        num_bits = [4]
        z_typical_4bit = [0.077, 1.013]                 # c1, c2 from the typical distribution (4bit)
        z_net_4bit = {'resnet20': [0.107, 0.881]}       # c1, c2 from the specifical models

        w_mean = self.weight.mean()
        weight_c = self.weight - w_mean
        q_scale = get_scale(self.weight, z_typical_4bit).item()

        weight_th = weight_c.clamp(-q_scale, q_scale)

        weight_th = q_scale * weight_th / 2 / torch.max(torch.abs(weight_th)) + q_scale / 2

        scale, zero_point = quantizer(num_bits[0], 0, abs(q_scale))
        weight_th = STEQuantizer.apply(weight_th, scale, zero_point, True, False)

        weight_q = 2 * weight_th - q_scale + w_mean

        q_levels = torch.unique(weight_q)
        idx = np.argsort(abs(q_levels.cpu().numpy()))
        
        weight_q[weight_q==q_levels[7]] = 0.
        weight_q[weight_q==q_levels[8]] = 0.

        output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output

class int_quant_func(torch.autograd.Function):
    def __init__(self, nbit):
        super(int_quant_func, self).__init__()
        self.nbit = nbit
    
    def forward(self, input):
        self.save_for_backward(input)

        z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}                 # c1, c2 from the typical distribution (4bit)
        z_net_4bit = {'resnet20': [0.107, 0.881]}                                   # c1, c2 from the specifical models

        alpha_w = get_scale(input, z_typical['4bit']).item()
        # print(f'alpha_w={alpha_w}')
        output = input.clamp(-alpha_w, alpha_w)

        # scale, zero_point = quantizer(self.nbit, -abs(alpha_w), abs(alpha_w))
        scale, zero_point = symmetric_linear_quantization_params(self.nbit, abs(alpha_w), restrict_qrange=True)
        # output = STEQuantizer.apply(output, scale, zero_point, True, False)
        output = STEQuantizer_weight.apply(output, scale, zero_point, True, False)
        return output
    
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class int_conv2d(nn.Conv2d):
    def forward(self, input):
        w_mean = self.weight.mean()
        weight_c = self.weight - w_mean

        weight_q = int_quant_func(nbit=4)(weight_c)
        
        q_levels = torch.unique(weight_q)
        
        # print(f'q_levels: {q_levels}')

        output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class sawb_tern_Conv2d(nn.Conv2d):

    def forward(self, input):
        z_typical_tern = [2.587, 1.693]                 # c1, c2 from the typical distribution (tern)
        quan_th = get_scale(self.weight, z_typical_tern)

        weight = sawb_ternFunc(th=quan_th)(self.weight)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output 
