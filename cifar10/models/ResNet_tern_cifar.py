import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .quant import clamp_conv2d, ClippedReLU, conv2d_Q_fn, learn_clp
import math

class _pruneFunc_mask(torch.autograd.Function):

    def __init__(self, tfactor):
        super(_pruneFunc_mask,self).__init__()
        self.tFactor = tfactor

    def forward(self, input, weight_zero, weight_keep):
        self.save_for_backward(input)
        input_keep = input[weight_keep.tolist(), :, :, :]
        max_w = input_keep.abs().max()
        self.th = self.tFactor*max_w #threshold

        self.W = input_keep[input_keep.ge(self.th)+input_keep.le(-self.th)].abs().mean()

        output = input.clone().zero_()

        output[input.ge(self.th)] = self.W
        output[input.lt(-self.th)] = -self.W

        output[weight_zero.tolist(), :, :, :] = 0.0

        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        # input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        #grad_input[weight_mask] = 0
        return grad_input, None, None


class _filterternFunc(torch.autograd.Function):

    def __init__(self, tfactor):
        super(_filterternFunc,self).__init__()
        self.tFactor = tfactor

    def forward(self, input):
        self.save_for_backward(input)
       
        # set the norm-rank x ration filters to zero

        weight_copy = input.data.abs().clone()
        L1_norm = torch.sum(weight_copy,(1,2,3))
        _,arg_max = torch.sort(L1_norm, descending=True)
        num_keep = int(weight_copy.shape[0] * 0.7)
        w_keep = arg_max[:num_keep]
        w_zeros = arg_max[num_keep:]
        weight_copy[w_zeros.tolist(), :, :, :] = 0.0

        input_keep = weight_copy[w_keep.tolist(), :, :, :]

        max_w = input_keep.abs().max()
        # self.th = 0.0001
        self.th = self.tFactor*max_w #threshold

        self.W = input_keep[input_keep.ge(self.th)+input_keep.le(-self.th)].abs().mean()
        weight_copy[weight_copy.ge(self.th)] = self.W
        weight_copy[weight_copy.lt(-self.th)] = -self.W
 
        return weight_copy

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input, None

class _quanFunc_mask(torch.autograd.Function):

    def __init__(self, tfactor):
        super(_quanFunc_mask,self).__init__()
        self.tFactor = tfactor

    def forward(self, input, weight_idx):
        self.save_for_backward(input)
        input_keep = input[weight_idx.tolist(), :, :, :]
        max_w = input_keep.abs().max()
        self.th = self.tFactor*max_w #threshold

        output = input.clone().zero_()
        self.W = input_keep[input_keep.ge(self.th)+input_keep.le(-self.th)].abs().mean()
        output[input.ge(self.th)] = self.W
        output[input.lt(-self.th)] = -self.W

        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        # input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        #grad_input[weight_mask] = 0
        return grad_input, None


class _quanFunc(torch.autograd.Function):

    def __init__(self, tfactor):
        super(_quanFunc,self).__init__()
        self.tFactor = tfactor

    def forward(self, input):
        self.save_for_backward(input)
        # mean_w = input.abs().mean()
        max_w = input.abs().max()
        # self.th = 0.0001
        self.th = self.tFactor*max_w #threshold

        # self.th = self.tFactor*max_w #threshold
        output = input.clone().zero_()
        self.W = input[input.ge(self.th)+input.le(-self.th)].abs().mean()
        output[input.ge(self.th)] = self.W
        output[input.lt(-self.th)] = -self.W

        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class quanConv2d(nn.Conv2d):

    def forward(self, input):
        tfactor_list = [0.05]
        weight = _quanFunc(tfactor=tfactor_list[0])(self.weight)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output 

class filterternConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(filterternConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation,
                                          groups=groups, bias=bias)
        self.get_weight_idx()

    def forward(self, input):
        tfactor_list = [0.05]
        # using swp to prune
        weight = _quanFunc_mask(tfactor=tfactor_list[0])(self.weight, self.w_idx)
        # directly prune
        # weight = _pruneFunc_mask(tfactor=tfactor_list[0])(self.weight, self.w_zero, self.w_idx)

        # weight = _quanFunc(tfactor=tfactor_list[0])(self.weight)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output 

    def get_weight_idx(self):
        '''
        Filter wise norm rank purning 
        This function get the weight mask when load the pretrained model,
        thus all the weights below the preset threshold will be considered
        as the pruned weights, and the returned weight index will be used
        for gradient masking.
        '''
        tfactor_list = [0.05]

        weight = _quanFunc(tfactor=tfactor_list[0])(self.weight)
        weight_copy = weight.data.abs().clone()

        # weight_copy = self.weight.data.abs().clone()

        #L2

        # weight_vec = weight_copy.view(weight_copy.size()[0], -1)
        # L2_norm = torch.norm(weight_vec, 2, 1)
        # _,arg_max = torch.sort(L2_norm)

        #L1
        L1_norm = torch.sum(weight_copy,(1,2,3))
        _,arg_max = torch.sort(L1_norm, descending=True)
        out_channels = weight_copy.shape[0]
        num_keep = int(16 * 0.7) * int(out_channels/16)
        #num_keep = int(out_channels * 0.9)
        # print(num_keep)
        # idx = arg_max[:num_keep]
        # self.w_idx, _ = idx.sort()
        self.w_idx = arg_max[:num_keep]
        self.w_zero = arg_max[num_keep:]
        return arg_max[num_keep:]

        # self.w_idx = arg_max[:num_keep]
        # self.w_zero = arg_max[num_keep:]
        # return arg_max[num_keep:]


class _quanLinear_mask(torch.autograd.Function):

    def __init__(self, tfactor):
        super(_quanLinear_mask,self).__init__()
        self.tFactor = tfactor

    def forward(self, input, weight_idx):
        self.save_for_backward(input)
        # input_keep = input
        input_keep = input[ :, weight_idx.tolist()]
        max_w = input_keep.abs().max()
        self.th = self.tFactor*max_w #threshold

        output = input.clone().zero_()
        self.W = input_keep[input_keep.ge(self.th)+input_keep.le(-self.th)].abs().mean()
        output[input.ge(self.th)] = self.W
        output[input.lt(-self.th)] = -self.W

        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        # input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        #grad_input[weight_mask] = 0
        return grad_input, None

class _pruneLinear_mask(torch.autograd.Function):

    def __init__(self, tfactor):
        super(_pruneLinear_mask,self).__init__()
        self.tFactor = tfactor

    def forward(self, input, weight_zero, weight_keep):
        self.save_for_backward(input)
        # input_keep = input
        input_keep = input[ :, weight_keep.tolist()]
        max_w = input_keep.abs().max()
        self.th = self.tFactor*max_w #threshold

        self.W = input_keep[input_keep.ge(self.th)+input_keep.le(-self.th)].abs().mean()

        output = input.clone().zero_()

        output[input.ge(self.th)] = self.W
        output[input.lt(-self.th)] = -self.W

        output[:, weight_zero.tolist()]

        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        # input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        #grad_input[weight_mask] = 0
        return grad_input, None, None

class quanLinear(nn.Linear):

    def forward(self, input):
        tfactor_list = [0.05]
        weight = _quanFunc(tfactor=tfactor_list[0])(self.weight)
        output = F.linear(input, weight, self.bias)

        return output

class pruneLinear(nn.Linear):
    def __init__(self, input_features, output_features, bias=True):
        super(pruneLinear, self).__init__(input_features, output_features)
        self.input_features = input_features
        self.output_features = output_features

        self.get_weight_idx()

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        tfactor_list = [0.05]

        #using swp prune
        weight = _quanLinear_mask(tfactor=tfactor_list[0])(self.weight, self.w_idx)
        # directly prune
        # weight = _pruneLinear_mask(tfactor=tfactor_list[0])(self.weight, self.w_zero, self.w_idx)

        output = F.linear(input, weight, self.bias)

        return output


    def get_weight_idx(self):
        '''
        Filter wise norm rank purning 
        This function get the weight mask when load the pretrained model,
        thus all the weights below the preset threshold will be considered
        as the pruned weights, and the returned weight index will be used
        for gradient masking.
        '''
        tfactor_list = [0.05]

        weight = _quanFunc(tfactor=tfactor_list[0])(self.weight)
        weight_copy = weight.data.abs().clone()
        # weight_copy = self.weight.data.abs().clone()

        #L2

        # weight_vec = weight_copy.view(weight_copy.size()[0], -1)
        # L2_norm = torch.norm(weight_vec, 2, 1)
        # _,arg_max = torch.sort(L2_norm)

        #L1
        L1_norm = torch.sum(weight_copy,0)
        
        _,arg_max = torch.sort(L1_norm,  descending=True)
        in_channels = weight_copy.shape[1]
        num_keep = int(16 * 0.7) * int(in_channels/16)
        #num_keep = int(out_channels * 0.9)
        # print(num_keep)

        # idx = arg_max[:num_keep]
        # self.w_idx, _ = idx.sort()
        self.w_idx = arg_max[:num_keep]
        self.w_zero = arg_max[num_keep:]
        return arg_max[num_keep:]

        # self.w_idx = arg_max[:num_keep]
        # self.w_zero = arg_max[num_keep:]

        # return arg_max[num_keep:]

class DownsampleA(nn.Module):

  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__()
    assert stride == 2
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

  def forward(self, x):
    x = self.avg(x)
    return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
  expansion = 1
  """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(ResNetBasicblock, self).__init__()

    # self.conv_a = quanConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  # aaai ternary
    # self.conv_a = sawb_tern_Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # sawb ternary
    # self.conv_a = clamp_conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  # quantization
    self.conv_a = learn_clp(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)   
    # self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)   # full precision
    self.bn_a = nn.BatchNorm2d(planes)

    # self.conv_b = quanConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  # aaai ternary
    # self.conv_b = sawb_tern_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  # sawb ternary
    # self.conv_b = clamp_conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  # quantization
    self.conv_b = learn_clp(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    # self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # full precision
    self.bn_b = nn.BatchNorm2d(planes)

    # self.relu1 = ClippedReLU(num_bits=4, alpha=10, inplace=True)    # Clipped ReLU function 4 - bits
    # self.reul2 = ClippedReLU(num_bits=4, alpha=10, inplace=True)    # Clipped ReLU function 4 - bits
    self.downsample = downsample

  def forward(self, x):
    residual = x

    basicblock = self.conv_a(x)
    basicblock = self.bn_a(basicblock)
    basicblock = F.relu(basicblock, inplace=True)
    # basicblock = self.relu(basicblock)

    basicblock = self.conv_b(basicblock)
    basicblock = self.bn_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    return self.relu2(residual + basicblock)


class CifarResNet(nn.Module):
  """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
  def __init__(self, block, depth, num_classes):
    """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
    super(CifarResNet, self).__init__()

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
    layer_blocks = (depth - 2) // 6
    print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))
    self.num_classes = num_classes
    self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

    # self.conv_1_3x3 = quanConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn_1 = nn.BatchNorm2d(16)

    self.inplanes = 16
    self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
    self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
    self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
    self.avgpool = nn.AvgPool2d(8)
    self.classifier = nn.Linear(64*block.expansion, num_classes)

    # self.classifier = quanLinear(64*block.expansion, num_classes)
    # self.relu = ClippedReLU(num_bits=4, alpha=10, inplace=True)    # Clipped ReLU function 4 - bits    
 
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv_1_3x3(x)
    x = F.relu(self.bn_1(x), inplace=True)
    # x = self.relu(self.bn_1(x))
    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return self.classifier(x)


def tern_resnet20(num_classes=10):
  """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 20, num_classes)
  return model


def tern_resnet32(num_classes=10):
  """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 32, num_classes)
  return model


def tern_resnet44(num_classes=10):
  """Constructs a ResNet-44 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 44, num_classes)
  return model


def tern_resnet56(num_classes=10):
  """Constructs a ResNet-56 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 56, num_classes)
  return model

def tern_resnet110(num_classes=10):
  """Constructs a ResNet-110 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 110, num_classes)
  return model

