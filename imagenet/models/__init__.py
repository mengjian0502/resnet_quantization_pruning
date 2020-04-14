
#### Models for ImageNet ############
from .alexnet_vanilla import alexnet_vanilla
from .alexnet_quan import tern_alexnet_ff_lf, tern_alexnet_fq_lq

from .resnet_vanilla import resnet18
from .ResNet_tern import resnet18b_ff_lf_tex1, resnet18b_fq_lq_tex1, resnet18b_ff_lf_w4_a4_tex1
from .ResNet_tern import resnet34b_ff_lf_tex1, resnet34b_fq_lq_tex1
from .ResNet_tern import resnet50b_ff_lf_tex1, resnet50b_fq_lq_tex1
from .ResNet_tern import resnet101b_ff_lf_tex1, resnet101b_fq_lq_tex1

from .ResNet_REL_tex2 import resnet18b_fq_lq_tern_tex_2

