"""
quant_layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .quantizer import *

__all__ = ['ClippedReLU', 'clamp_conv2d', 'sawb_tern_Conv2d']


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

class learn_clp(nn.Conv2d):

    def forward(self, input):
        beta = 5.0
        beta = nn.Parameter(torch.Tensor([beta]))
        num_bits = [4]

        w_l = self.weight

        w_mean = self.weight.mean()                     # mean of the weight
        w_l = w_l - w_mean                              # center the weights

        w_l = beta * torch.tanh(w_l)       
        w_l = beta * w_l / 2 / torch.max(torch.abs(w_l)) + beta / 2
        scale, zero_point = quantizer(num_bits[0], 0, abs(beta))

        w_l = STEQuantizer.apply(w_l, scale, zero_point, True, False)

        w_q = 2 * w_l - beta + w_mean

        output = F.conv2d(input, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output
    
class clamp_conv2d(nn.Conv2d):

    def forward(self, input):
        
        num_bits = [4]
        z_typical_4bit = [0.077, 1.013]                 # c1, c2 from the typical distribution (4bit)
        z_net_4bit = {'resnet20': [0.107, 0.881]}       # c1, c2 from the specifical models

        w_mean = self.weight.mean()
        weight_c = self.weight - w_mean
        q_scale = get_scale(self.weight, z_net_4bit['resnet20']).item()

        weight_th = weight_c.clamp(-q_scale, q_scale)

        weight_th = q_scale * weight_th / 2 / torch.max(torch.abs(weight_th)) + q_scale / 2

        scale, zero_point = quantizer(num_bits[0], 0, abs(q_scale))
        weight_th = STEQuantizer.apply(weight_th, scale, zero_point, True, False)

        weight_q = 2 * weight_th - q_scale + w_mean

        output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output

class sawb_tern_Conv2d(nn.Conv2d):

    def forward(self, input):
        z_typical_tern = [2.587, 1.693]                 # c1, c2 from the typical distribution (tern)
        quan_th = get_scale(self.weight, z_typical_tern)

        weight = sawb_ternFunc(th=quan_th)(self.weight)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output 
