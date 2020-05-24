###cifar 10
from .ResNet_tern_cifar import tern_resnet20, tern_resnet32, tern_resnet44, tern_resnet56, tern_resnet110
from .ResNet_tern_cifar_prob import tern_resnet20_w2_a2_zeroskp
from .ResNet_tern_cifar_adcinf import adc_resnet20


from .ResNet_cifar_vanilla import resnet20, resnet32, resnet44, resnet56, resnet110
from .preresnet import preresnet20
from .vgg16 import vgg16bn_cifar
from .vgg import vgg7, vgg7_quant

#### Models for ImageNet ############
from .alexnet_vanilla import alexnet_vanilla
from .alexnet_quan import tern_alexnet_ff_lf, tern_alexnet_fq_lq

from .resnet_vanilla import resnet18
from .ResNet_tern import resnet18b_ff_lf_tex1, resnet18b_fq_lq_tex1
from .ResNet_tern import resnet34b_ff_lf_tex1, resnet34b_fq_lq_tex1
from .ResNet_tern import resnet50b_ff_lf_tex1, resnet50b_fq_lq_tex1
from .ResNet_tern import resnet101b_ff_lf_tex1, resnet101b_fq_lq_tex1

from .ResNet_REL_tex2 import resnet18b_fq_lq_tern_tex_2