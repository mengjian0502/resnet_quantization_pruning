"""
Inference with different ADC precisions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch._jit_internal import weak_script_method
from .utee import wage_initializer,wage_quantizer
import numpy as np
from .quantizer import *
from .quant_layer import *

class Qconv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                col_size=16, group_size=8, wl_input=8, wl_weight=8,inference=0,cellBit=1,ADCprecision=5):
        super(Qconv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)

        self.col_size = col_size
        self.group_size = group_size
        self.wl_input = wl_input
        self.inference = inference
        self.wl_weight = wl_weight
        self.cellBit = cellBit
        self.ADCprecision = ADCprecision
        self.act_alpha = 1.
        self.layer_idx = 0
        self.iter = 0

    @weak_script_method
    def forward(self, input):

        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        weight_c = self.weight - self.weight.mean()
        weight_q, alpha_w, w_scale = odd_symm_quant(weight_c, nbit=bitWeight, dequantize=True, posQ=False)                                               # quantize the input weights
        weight_int, _, _ = odd_symm_quant(weight_c, nbit=bitWeight, dequantize=False, posQ=True)                                                         # quantize the input weights to positive integer
        
        output_original = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)


        if self.inference == 1:
            num_sub_groups = self.col_size // self.group_size

            output = torch.zeros_like(output_original)
            # del output_original
            cellRange = 2**self.cellBit   # cell precision is 2

            # loop over the group of rows
            for ii in range(num_sub_groups):
                mask = torch.zeros_like(weight_q)
                for jj in np.arange(0, weight_q.size(1)//self.group_size, num_sub_groups):
                    mask[:, (ii+jj)*self.group_size:(ii+jj+1)*self.group_size,:,:] = 1                                                                  # turn on the corresponding rows.

                # print(f'Layer: {self.layer_idx}, 4 bit input activation: {torch.unique(input)} | Num levels: {len(torch.unique(input))}')

                inputQ, act_scale = activation_quant(input, nbit=bitActivation, sat_val=self.act_alpha, dequantize=False)
                outputIN = torch.zeros_like(output)

                for z in range(bitActivation):
                    inputB = torch.fmod(inputQ, 2)
                    inputQ = torch.round((inputQ-inputB)/2)
                    outputP = torch.zeros_like(output)                    
                    X_decimal = weight_int*mask                                                                                                              # multiply the quantized integer weight with the corresponding mask
                    outputD = torch.zeros_like(output)
                    outputDiff = torch.zeros_like(output)

                    dummyP = torch.zeros_like(weight_q)
                    dummyP[:,:,:,:] = 7

                    for k in range(int(bitWeight/self.cellBit)):
                        remainder = torch.fmod(X_decimal, cellRange)*mask
                        X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                        
                        outputPartial= F.conv2d(inputB, remainder*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)                      # Binarized convolution

                        scaler = cellRange**k
                        outputP = outputP + outputPartial*scaler

                    outputDummyPartial = F.conv2d(inputB, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups) 
                    output_diff = outputP - outputDummyPartial
                    output_diff_quant = wage_quantizer.LinearQuantizeOut(output_diff, self.ADCprecision)
                    outputDiff = outputDiff + output_diff_quant

                    scalerIN = 2**z
                    outputIN = outputIN + outputDiff*scalerIN
                output = output + outputIN/act_scale                                                                                                       # dequantize it back    
            output = output/w_scale

            # print(f'Layer:{self.layer_idx} alpha: {self.act_alpha}')

            # plt.figure(figsize=(8,6))
            # sns.distplot(output_original.cpu().numpy().flatten(), label='Original output')
            # sns.distplot(output.cpu().numpy().flatten(), label='Layer output')
            # plt.legend(loc='best')
            # plt.savefig(f'./layer_plot/output_dist_layer{self.layer_idx}.png')
        return output         

class QLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, col_size=16, group_size=8, 
                wl_input=8,wl_activate=8,wl_error=8,wl_weight= 8,inference=0,onoffratio=10,cellBit=1,ADCprecision=5):
        super(QLinear, self).__init__(in_features=in_channels, out_features=out_channels, bias=bias)

        self.col_size = col_size
        self.group_size = group_size
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.wl_input = wl_input
        self.inference = inference
        self.wl_weight = wl_weight
        self.cellBit = cellBit
        self.ADCprecision = ADCprecision
        self.act_alpha = 1.

    @weak_script_method

    def forward(self, input):
        
        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        weight_c = self.weight - self.weight.mean()
        weight_q, alpha_w, w_scale = odd_symm_quant(weight_c, nbit=bitWeight, dequantize=True, posQ=False)                                               # quantize the input weights
        weight_int, _, _ = odd_symm_quant(weight_c, nbit=bitWeight, dequantize=False, posQ=True)        

        output_original = F.linear(input, weight_q, self.bias)

        if self.inference == 1:
            output = torch.zeros_like(output_original)
            del output_original
            cellRange = 2**self.cellBit   # cell precision is 2

            # dummy crossbar
            dummyP = torch.zeros_like(weight_q)
            dummyP[:,:] = 7

            # since fc weight size is relatively small => Directly generate output
            mask=torch.zeros_like(weight_q)
            mask[:, :] = 1

            inputQ, act_scale = activation_quant(input, nbit=bitActivation, sat_val=self.act_alpha, dequantize=False)                          # quantize the input activation to the integer value
            outputIN = torch.zeros_like(output)

            for z in range(bitActivation):
                inputB = torch.fmod(inputQ, 2)
                inputQ = torch.round((inputQ-inputB)/2)

                # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                X_decimal = weight_int * mask                                                                                   # multiply the quantized integer weight with the corresponding mask
                outputP = torch.zeros_like(output)
                outputD = torch.zeros_like(output)
                outputDiff = torch.zeros_like(output)

                for k in range (int(bitWeight/self.cellBit)):
                    remainder = torch.fmod(X_decimal, cellRange)*mask
                    X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask

                    outputPartial= F.linear(inputB, remainder*mask, self.bias)

                    # Add ADC quanization effects here !!!
                    # outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                    # print(self.ADCprecision)

                    scaler = cellRange**k
                    outputP = outputP + outputPartial*scaler
                
                outputDummyPartial= F.linear(inputB, dummyP*mask, self.bias)
                output_diff = outputP - outputDummyPartial
                output_diff_quant = wage_quantizer.LinearQuantizeOut(output_diff, self.ADCprecision)
                # print(f'Linear layer outputDummyPartial: {torch.unique(outputDummyPartial)}')
                # outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                # outputD = outputD + outputDummyPartialQ
                outputDiff = outputDiff + output_diff_quant

                scalerIN = 2**z
                outputIN = outputIN + outputDiff*scalerIN    
            output = output + outputIN/act_scale
        output = output/w_scale
        return output