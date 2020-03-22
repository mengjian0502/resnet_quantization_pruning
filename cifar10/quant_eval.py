"""
ResNet + CIFAR10: Evaluate the model
"""

import os
import sys
import shutil
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from collections import OrderedDict
from utils_.utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
from utils_.reorganize_param import reorganize_param
from utils_.model_summary import summary
import csv

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import models
from models.ResNet_tern_cifar import filterternConv2d, pruneLinear, quanConv2d, _quanFunc
from models.ResNet_cifar_vanilla import filterpruneConv2d, pLinear
from logger import Logger