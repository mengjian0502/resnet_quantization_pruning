"""
quant_layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .quantizer import *


def odd_symm_quant(input, nbit, dequantize=True, posQ=False):
    # z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}

    # alpha_w = get_scale(input, z_typical[f'{nbit}bit']).item()
    alpha_w = 2 * input.abs().mean()
    output = input.clamp(-alpha_w, alpha_w)

    if posQ:
        output = output + alpha_w

    scale, zero_point = symmetric_linear_quantization_params(nbit, abs(alpha_w), restrict_qrange=True)

    output = linear_quantize(output, scale, zero_point)
    
    if dequantize:
        output = linear_dequantize(output, scale, zero_point)

    return output, alpha_w, scale

def activation_quant(input, nbit, sat_val, dequantize=True):
    with torch.no_grad():
        scale, zero_point = quantizer(nbit, 0, sat_val)
    
    output = linear_quantize(input, scale, zero_point)

    if dequantize:
        output = linear_dequantize(output, scale, zero_point)

    return output, scale

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

class zero_skp_quant(torch.autograd.Function):
    def __init__(self, nbit, coef, group_ch):
        super(zero_skp_quant, self).__init__()
        self.nbit = nbit
        self.coef = coef
        self.group_ch = group_ch
    
    def forward(self, input):
        self.save_for_backward(input)

        # alpha_w_original = get_scale(input, z_typical[f'{int(self.nbit)}bit'])
        alpha_w_original = get_scale_2bit(input)
        interval = 2*alpha_w_original / (2**self.nbit - 1) / 2
        self.th = self.coef * interval

        cout = input.size(0)
        cin = input.size(1)
        kh = input.size(2)
        kw = input.size(3)
        num_group = (cout * cin) // self.group_ch

        w_t = input.view(num_group, self.group_ch*kh*kw)

        grp_values = w_t.norm(p=2, dim=1)                                               # L2 norm
        mask_1d = grp_values.gt(self.th*self.group_ch*kh*kw).float()
        # mask_dropout = torch.rand_like(mask_1d_small).lt(self.prob).float()

        # apply the probablistic sampling:
        # mask_1d = 1 - mask_1d_small * mask_dropout
        mask_2d = mask_1d.view(w_t.size(0),1).expand(w_t.size()) 

        w_t = w_t * mask_2d

        non_zero_idx = torch.nonzero(mask_1d).squeeze(1)                             # get the indexes of the nonzero groups
        non_zero_grp = w_t[non_zero_idx]                                             # what about the distribution of non_zero_group?
        
        weight_q = non_zero_grp.clone()
        alpha_w = get_scale_2bit(weight_q)

        weight_q[non_zero_grp.ge(alpha_w - alpha_w/3)] = alpha_w
        weight_q[non_zero_grp.lt(-alpha_w + alpha_w/3)] = -alpha_w
        
        weight_q[non_zero_grp.lt(alpha_w - alpha_w/3)*non_zero_grp.ge(0)] = alpha_w/3
        weight_q[non_zero_grp.ge(-alpha_w + alpha_w/3)*non_zero_grp.lt(0)] = -alpha_w/3

        w_t[non_zero_idx] = weight_q
        
        output = w_t.clone().resize_as_(input)
        return output

    def backward(self, grad_output):
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
        # print(f'ClippedRELU: input mean: {input.mean()} | input std: {input.std()}')
        input = F.relu(input)
        input = torch.where(input < self.alpha, input, self.alpha)
        
        with torch.no_grad():
            scale, zero_point = quantizer(self.num_bits, 0, self.alpha)
        input = STEQuantizer.apply(input, scale, zero_point, self.dequantize, self.inplace)
        return input

class ClippedReLU_dummy(nn.Module):
    def __init__(self, num_bits, alpha=8.0, inplace=False, dequantize=True):
        super(ClippedReLU_dummy, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]))     
        self.num_bits = num_bits
        self.inplace = inplace
        self.dequantize = dequantize
        
    def forward(self, input):
        input = F.relu(input)
        return input
    
class clamp_conv2d(nn.Conv2d):

    def forward(self, input):
        
        num_bits = [2]
        z_typical = {'2bit': [0.311, 0.678], '4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}

        # w_mean = self.weight.mean()

        weight_c = self.weight

        q_scale = get_scale(self.weight, z_typical[f'{num_bits[0]}bit']).item()

        weight_th = weight_c.clamp(-q_scale, q_scale)

        weight_th = q_scale * weight_th / 2 / torch.max(torch.abs(weight_th)) + q_scale / 2

        scale, zero_point = quantizer(num_bits[0], 0, abs(q_scale))
        weight_th = STEQuantizer.apply(weight_th, scale, zero_point, True, False)

        weight_q = 2 * weight_th - q_scale

        output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output

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

class int_quant_func(torch.autograd.Function):
    def __init__(self, nbit, alpha_w, restrictRange=True):
        super(int_quant_func, self).__init__()
        self.nbit = nbit
        self.restrictRange = restrictRange
        self.alpha_w = alpha_w

    def forward(self, input):
        self.save_for_backward(input)

        # z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}                 # c1, c2 from the typical distribution (4bit)
        # z_net_4bit = {'resnet20': [0.107, 0.881]}                                   # c1, c2 from the specifical models

        # alpha_w = get_scale(input, z_typical['4bit']).item()
        output = input.clamp(-self.alpha_w.item(), self.alpha_w.item())

        scale, zero_point = symmetric_linear_quantization_params(self.nbit, self.alpha_w, restrict_qrange=self.restrictRange)
        output = STEQuantizer_weight.apply(output, scale, zero_point, True, False, self.nbit, self.restrictRange)

        return output

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class int_conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False, nbit=4):
        super(int_conv2d, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.nbit = nbit

    def forward(self, input):
        # w_mean = self.weight.mean()
        weight_c = self.weight.clone()
        
        self.alpha_w = 2 * self.weight.abs().mean()
        weight_q = int_quant_func(nbit=self.nbit, alpha_w=self.alpha_w, restrictRange=True)(weight_c)
        output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output
    
    def extra_repr(self):
        return super(int_conv2d, self).extra_repr() + ', nbit={}'.format(self.nbit)


class int_linear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, nbit=8):
        super(int_linear, self).__init__(in_features=in_channels, out_features=out_channels, bias=bias)
        self.nbit=nbit
    
    def forward(self, input):
        w_mean = self.weight.mean()
        weight_c = self.weight.clone()

        self.alpha_w = 2 * self.weight.abs().mean()
    
        if self.nbit == 2:
            alpha_w = get_scale_2bit(self.weight)
            weight_q = sawb_w2_Func(alpha=self.alpha_w)(weight_c)
        else:
            weight_q = int_quant_func(nbit=self.nbit, alpha_w=self.alpha_w)(weight_c)
        
        output = F.linear(input, weight_q, self.bias)
        return output

    def extra_repr(self):
        return super(int_linear, self).extra_repr() + ', nbit={}'.format(self.nbit)

class sawb_tern_Conv2d(nn.Conv2d):

    def forward(self, input):
        z_typical_tern = [2.587, 1.693]                 # c1, c2 from the typical distribution (tern)
        quan_th = get_scale(self.weight, z_typical_tern)

        weight = sawb_ternFunc(th=quan_th)(self.weight)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output

class sawb_w2_Conv2d(nn.Conv2d):

    def forward(self, input):
        alpha_w = get_scale_2bit(self.weight)

        weight = sawb_w2_Func(alpha=alpha_w)(self.weight)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output


class zero_grp_skp_quant(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False, nbit=2, coef=0.05, skp_group=8):
        super(zero_grp_skp_quant, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.nbit = nbit
        self.coef = coef
        self.skp_group = skp_group

    def forward(self, input):
        weight = self.weight

        weight_q = zero_skp_quant(nbit=self.nbit, coef=self.coef, group_ch=self.skp_group)(weight)
        output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output
    
    def extra_repr(self):
        return super(zero_grp_skp_quant, self).extra_repr() + ', nbit={}, coef={}, skp_group={}'.format(
                self.nbit, self.coef, self.skp_group)

