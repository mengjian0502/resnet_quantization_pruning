import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def weight_tern(fp_w, t_factor):
    fp_w = torch.from_numpy(fp_w)
    mean_w = fp_w.abs().mean()
    max_w = fp_w.abs().max()
    th = t_factor*max_w

    output = fp_w.clone().zero_()
    W = fp_w[fp_w.ge(th)+fp_w.le(-th)].abs().mean()
    output[fp_w.ge(th)] = 1
    output[fp_w.lt(-th)] = -1
    return output.numpy()

def to_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val) if is_scalar else sat_val.clone().detach()
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out

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


class STEQuantizer(torch.autograd.Function):
    """
    Linear quantization + Straight through estimator (STE)
    """
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

def clamp_quant(alpha, x, nbit):
    _, x = to_tensor(x)
    w_mean = x.mean()
    weight_c = x - w_mean

    weight_th = weight_c.clamp(-alpha, alpha)

    weight_th = alpha * weight_th / 2 / torch.max(torch.abs(weight_th)) + alpha / 2

    scale, zero_point = quantizer(nbit, 0, abs(alpha))
    weight_th = STEQuantizer.apply(weight_th, scale, zero_point, True, False)

    weight_q = 2 * weight_th - alpha + w_mean
    return weight_q.numpy()

def first_order_func(z, input):
    """
    First order regression to compute the quantization scale
    """

    c1, c2 = 1 / z[0], z[1] / z[0]

    _, input = to_tensor(input)

    std = input.std()
    mean = input.abs().mean()
    
    q_scale = c1 * std - c2 * mean
    
    return q_scale

model_dir = './save/resnet18/sgd/resnet18_w4_a4_swpTrue_from_quant_lambda0.00025_g02/model_best.pth.tar'
model = torch.load(model_dir, map_location=torch.device('cpu'))
# model = torch.load(model_dir)
params = model['state_dict']
print(params)
counter = 0
overall = 0
num_one = 0
all_num_one = 0.0
all_num_group = 0.0
group_ch = 16
all_sparsity_mean = 0.0
all_sparsity_min = 1.0
all_num = 0.0
count_num_one = 0.0
t_factor = 0.05
tern = False
quant = True

z = [0.077, 1.013]
for k,v in params.items():
    if 'layer' in k and 'conv' in k:
        print(k)
        
        counter += 1
        conv_w = v 
        output = np.array(conv_w)
        cout = output.shape[0]
        cin = output.shape[1]
        kh = output.shape[2]
        kw = output.shape[3]
        
        if tern: 
            out_sparse = weight_tern(output, t_factor)
        elif quant:
            alpha_w = first_order_func(z, output)
            out_sparse = clamp_quant(alpha_w, output, 4)            
            print(out_sparse)
        else:
        # == set threshold for weight
            out_sparse = np.zeros(output.shape)
            out_sparse[np.absolute(output) > 1e-3] = output[np.absolute(output) > 1e-3]
        # ==
        
        #cal the whole number of weight in current layer
        count_num_layer = cout * cin * kh * kw
        all_num += count_num_layer
        #cal the nonzero weight in current layer
        count_one_layer = np.count_nonzero(out_sparse)
        count_num_one += count_one_layer
        
        w_t = out_sparse.reshape(cout, cin // group_ch, group_ch, kh, kw)
        num_group = (cout * cin) // group_ch
        w_t = w_t.reshape(num_group, group_ch, kh, kw)
        all_num_group += num_group
        sparsity_chall = np.zeros(num_group)
        num_one = 0
        for i in range(num_group):       
            sparsity_group = 1.0*(w_t[i].size - np.count_nonzero(w_t[i]))/w_t[i].size
            sparsity_chall[i] = sparsity_group
            if np.count_nonzero(w_t[i]) is 0:
                num_one += 1
        all_num_one += num_one 
        print('num of sparse group:{}'.format(num_one))
        print('num of group:{}'.format(num_group))

print("==== conclusion ==== ")
overall_sparsity = 1 - count_num_one/all_num
print('overall sparsity for all layer:{:.6f}'.format(overall_sparsity))
group_sparsity = all_num_one/all_num_group
print('group sparsity for all layer:{:.6f}'.format(group_sparsity))