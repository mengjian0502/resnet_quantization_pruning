"""
Inference with different ADC precisions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utee import wage_initializer,wage_quantizer
from torch._jit_internal import weak_script_method
import numpy as np
from .quantizer import *
from .quant_layer import *

class Qconv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, 
                col_size=16, group_size=8, wl_input=8,wl_activate=8,wl_error=8,wl_weight= 8,inference=0,onoffratio=10,cellBit=1,subArray=128,ADCprecision=5,vari=0)
        super(Qconv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)

        self.col_size = col_size
        self.group_size = group_size
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.wl_input = wl_input
        self.inference = inference
        self.wl_weight = wl_weight
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.act_alpha = 1.
    @weak_script_method
    def forward(self, input):

        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        weight_c = self.weight - self.weight.mean()
        weight_q, alpha_w, w_scale = odd_symm_quant(input, nbit=bitWeight, dequantize=True, posQ=False)                                               # quantize the input weights
        weight_int, _, _ = odd_symm_quant(input, nbit=bitWeight, dequantize=False, posQ=True)                                             # quantize the input weights to positive integer

        output_original = F.Conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if self.inference == 1:
            onoffratio = self.onoffratio
            upper = 1
            lower = 1/onoffratio
            num_sub_groups = self.col_size // self.group_size

            output = torch.zeros_like(outputOrignal)
            del outputOrignal
            cellRange = 2**self.cellBit   # cell precision is 4

            # dummy crossbar
            dummyP = torch.zeros_like(weight)
            dummyP[:,:,:,:] = (cellRange-1)*(upper+lower)/2

            # reshape the weight
            kw = weight_int.size(2)
            num_group = weight_int.size(0) * weight_int.size(1) // group_ch
            weight_int = weight_int.view(weight_int.size(0), weight_int.size(1) // self.col_size, self.col_size, kw, kw)
            weight_int = weight_int.view(num_group, self.col_size, kw, kw)                                                          # reshape the 4-D weight tensor to the column based shape

            # loop over the group of rows
            for ii in range(num_sub_groups):
                mask = torch.zeros_like(weight_q)
                mask[:, ii*self.group_size:(ii+1)*group_size,:,:] = 1                                                               # turn on the corresponding rows.

                # quantize the activation
                inputQ = activation_quant(input, nbit=bitActivation, sat_val=self.alpha, dequantize=False)                          # quantize the input activation to the integer value
                outputIN = torch.zeros_like(output)    

                for z in range(bitActivation):
                    inputB = torch.fmod(inputQ, 2)
                    inputQ = torch.round((inputQ-inputB)/2)
                    outputP = torch.zeros_like(output)

                    X_decimal = weight_int * mask                                                                                   # multiply the quantized integer weight with the corresponding mask
                    outputD = torch.zeros_like(output)                                                                              # dummy output

                    for k in range (int(bitWeight/self.cellBit)):
                        remainder = torch.fmod(X_decimal, cellRange)*mask
                        variation = np.random.normal(0, self.vari, list(weight.size())).astype(np.float32)                          # variation of the circuit
                        X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask

                        remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower                                                # weight cannot map to 0, but to Gmin
                        remainderQ = remainderQ + remainderQ*torch.from_numpy(variation).cuda()
                        
                        outputPartial= F.conv2d(inputB, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                        outputDummyPartial= F.conv2d(inputB, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

                        # Add ADC quanization effects here !!!
                        outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                        outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                        scaler = cellRange**k
                        outputSP = outputSP + outputPartialQ*scaler*2/(1-1/onoffratio)
                        outputD = outputD + outputDummyPartialQ*scaler*2/(1-1/onoffratio)
                        # !!! Important !!! the dummy need to be multiplied by a ratio
                        outputSP = outputSP - outputD  # minus dummy column
                        outputP = outputP + outputSP
                    scalerIN = 2**z
                    outputIN = outputIN + (outputP - outputD)*scalerIN
                output = output + outputIN/((2**bitActivation - 1)/self.alpha)                                                      # dequantize it back    
            output = output/w_scale
        return output         






                

