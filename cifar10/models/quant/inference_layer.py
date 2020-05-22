"""
Inference with different ADC precisions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utee import wage_initializer,wage_quantizer
from torch._jit_internal import weak_script_method
from quant import odd_symm_quant
import numpy as np


class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,logger = None,clip_weight = False,wage_init=False,quantize_weight= False,clip_output =False,quantize_output = False,
                 wl_input=8,wl_activate=8,wl_error=8,wl_weight=8,inference=0,onoffratio=10,cellBit=1,subArray=128,ADCprecision=5,vari=0,t=0,v=0,detect=0,target=0,debug=0, col_ch=16, group_ch=8, name='Qconv'):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.logger = logger
        self.clip_weight = clip_weight
        self.wage_init = wage_init
        self.quantize_weight = quantize_weight
        self.clip_output = clip_output
        self.debug = debug
        self.wl_weight = wl_weight
        self.quantize_output = quantize_output
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.wl_input = wl_input
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.name = name
        self.scale  = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0)
        self.col_ch = col_ch
        self.group_ch = group_ch

        @weak_script_method
        def forward(self, input):
            
            output_example = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)  # not the actual output!
            
            bitWeight = int(self.wl_weight)
            bitActivation = int(self.wl_input)    

            if self.inference == 1:
                # skip the retention

                onoffratio = self.onoffratio
                upper = 1
                lower = 1/onoffratio

                output = torch.zeros_like(output_example)
                del output_example
                cellRange = 2**self.cellBit     # in our case, cellBit=1

                # Now consider on/off ratio
                dummyP = torch.zeros_like(weight)
                dummyP[:,:,:,:] = (cellRange-1)*(upper+lower)/2

                # number of channel groups
                num_grp = self.weight.size(1) // self.col_ch
                num_sub_grp = self.col_ch // self.group_ch

                # quantize the weights
                _, X_decimal, _ = odd_symm_quant(self.weight, bitWeight)    # get the integer values
                X_decimal = X_decimal + (2**bitWeight - 2) // 2             # shift the integer values between [0, 2**bitWeight-1]
                
                # quantize the input
                # todo: send the alpha of activation into this layer

                for z in range(bitActivation):

                    for i in range(num_grp):
                        mask = torch.zeros_like(self.weight[:, (i*self.group_ch):((i+1)*self.group_ch), :, :])
                        for j in range(num_sub_grp):
                            mask[:, (j*self.group_ch):(j+1)*self.group_ch, :, :] = 1
                            X_decimal = X_decimal * mask
                            outputD = torch.zeros_like(output)

