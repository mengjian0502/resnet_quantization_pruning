"""
Statistics Aware Weight Bining (SAWB) + PACT implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def to_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val) if is_scalar else sat_val.clone().detach()
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out

def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Linearly quantize the input tensor based on scale and zero point.
    https://pytorch.org/docs/stable/quantization.html
    """
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(input * scale - zero_point)

def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale

def quantizer(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    scalar_min, sat_min = to_tensor(saturation_min)
    scalar_max, sat_max = to_tensor(saturation_max)
    is_scalar = scalar_min and scalar_max

    if scalar_max and not scalar_min:
        sat_max = sat_max.to(sat_min.device)
    elif scalar_min and not scalar_max:
        sat_min = sat_min.to(sat_max.device)

    if any(sat_min > sat_max):
        raise ValueError('saturation_min must be smaller than saturation_max')

    n = 2 ** num_bits - 1

    # Make sure 0 is in the range
    sat_min = torch.min(sat_min, torch.zeros_like(sat_min))
    sat_max = torch.max(sat_max, torch.zeros_like(sat_max))

    diff = sat_max - sat_min
    diff[diff == 0] = n

    scale = n / diff
    zero_point = scale * sat_min
    if integral_zero_point:
        zero_point = zero_point.round()
    if signed:
        zero_point += 2 ** (num_bits - 1)
    if is_scalar:
        return scale.item(), zero_point.item()
    return scale, zero_point
        

def get_scale_2bit(input):
    c1, c2 = 3.212, -2.178
    
    std = input.std()
    mean = input.abs().mean()
    
    q_scale = c1 * std + c2 * mean
    
    return q_scale 

def get_scale_tern(input):
    c1, c2 = 2.587, -1.693
    
    std = input.std()
    mean = input.abs().mean()
    
    q_scale = c1 * std + c2 * mean
    
    return q_scale

def get_scale_reg2(input):
    z1, z2, z3 = 0.37505558, -1.26150945, 2.30980168

    std = input.std()
    mean = input.abs().mean()
    
    a1, a2, a3 = z1, z2*mean, mean*(z3*mean - std)
    
    alpha_w = np.roots([a1, a2, a3]) 
    return np.real(alpha_w[0])

# def sawb_w2_Func(k, alpha):
#     class qfn(torch.autograd.Function):
#         @staticmethod
#         def forward(ctx, input):
#             if k == 32:
#                 out = input
#             elif k == 1:
#                 out = torch.sign(input)
#             else:
#                 self.save_for_backward(input)

#                 output = input.clone()
#                 output[input.ge((alpha - alpha/3)/2)] = alpha
#                 output[input.lt((-alpha + alpha/3)/2)] = -alpha
            
#                 output[input.lt((alpha - alpha/3)/2)*input.ge(0)] = alpha/3
#                 output[input.ge((-alpha + alpha/3)/2)*input.lt(0)] = -alpha/3
#             return output

#         @staticmethod
#         def backward(ctx, grad_output):
#             grad_input = grad_output.clone()
#             input, = self.saved_tensors
#             grad_input[input.ge(1)] = 0
#             grad_input[input.le(-1)] = 0
#             return grad_input
#     return qfn().apply

    
# class weight_quantize_fn(nn.Module):
#     def __init__(self, w_bit):
#         super(weight_quantize_fn, self).__init__()
#         assert w_bit <= 8 or w_bit == 32
#         self.w_bit = w_bit

#     def forward(self, x):
#         q_scale = get_scale_2bit(x)
#         self.uniform_q = sawb_w2_Func(self.w_bit, q_scale)
#         weight_q = self.uniform_q(x)

#         return weight_q
    

class STEQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)
        
        output = linear_quantize(input, scale, zero_point)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point)  
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight Through Estimator
        """
        
        return grad_output, None, None, None, None
    
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

# class SAWB_conv2d(nn.Conv2d):

#     def forward(self, input):
        
#         num_bits = [2]

#         q_scale = get_scale(self.weight)

#         weight_th = q_scale * torch.tanh(self.weight)

#         weight = q_scale * weight_th / 2 / torch.max(torch.abs(weight_th)) + q_scale / 2

#         scale, zero_point = quantizer(num_bits[0], 0, q_scale)
#         weight = STEQuantizer.apply(weight, scale, zero_point, True, False)

#         weight_q = 2 * weight - q_scale

#         output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
#         return output


# def SAWB_2bit_Conv2d(w_bit):
#   class SAWB_conv(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#       super(SAWB_conv, self).__init__(in_channels, out_channels, kernel_size, stride,
#                                      padding, dilation, groups, bias)
#       self.w_bit = w_bit
#       self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

#     def forward(self, input, order=None):
#       weight_q = self.quantize_fn(self.weight)
#       return F.conv2d(input, weight_q, self.bias, self.stride,
#                       self.padding, self.dilation, self.groups)

#   return SAWB_conv

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

class sawb_tern_Conv2d(nn.Conv2d):

    def forward(self, input):
        quan_th = get_scale_tern(self.weight)

        weight = sawb_ternFunc(th=quan_th)(self.weight)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output 

class sawb_w2_Conv2d(nn.Conv2d):

    def forward(self, input):
        alpha_w = get_scale_2bit(self.weight)
        # alpha_w = get_scale_reg2(self.weight)

        weight = sawb_w2_Func(alpha=alpha_w)(self.weight)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output 
        
