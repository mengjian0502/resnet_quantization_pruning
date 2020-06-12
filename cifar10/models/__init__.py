###cifar 10
from .ResNet_tern_cifar import tern_resnet20, tern_resnet32, tern_resnet44, tern_resnet56, tern_resnet110
from .ResNet_tern_cifar_prob import tern_resnet20_w2_a2_zeroskp
from .ResNet_tern_cifar_adcinf import adc_resnet20
from .ResNet_tern_cifar_learnableQ import resnet20_lq


from .ResNet_cifar_vanilla import resnet20, resnet32, resnet44, resnet56, resnet110
from .preresnet import preresnet20
from .vgg16 import vgg16bn_cifar
from .vgg import vgg7, vgg7_quant
from .vgg_adcinfer import vgg7_adc

from .resnet_vanilla import resnet18
