from __future__ import division
from __future__ import absolute_import

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
from models.quant import int_conv2d, int_quant_func, Qconv2d, QLinear
from torchsummary import summary
import csv

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import models
from logger import Logger

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(description='Training network for image classification',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path', default='/home/elliot/data/pytorch/svhn/',
                    type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist'],
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='lbcnn', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
# Checkpoints
parser.add_argument('--save_path', type=str, default='./save/',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int,
                    metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate',
                    action='store_true', help='evaluate model on validation set')
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                    help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--model_only', dest='model_only', action='store_true',
                    help='only save the model without external utils_')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='device range [0,ngpu-1]')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default=5000, help='manual seed')

# Inference with ADC
parser.add_argument('--adc_infer', action='store_true', help='True for adc inference')
parser.add_argument('--col_size', type=int, default=16, help='Number of rows per column')
parser.add_argument('--group_size', type=int, default=16, help='Number of channels for each iso-group')
parser.add_argument('--ADCprecision', type=int, default=5, help='ADC precision of the RRAM')
parser.add_argument('--cell_bit', type=int, default=2, help='precision of each memory cell')
##########################################################################

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    # make only device #gpu_id visible, then
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True


###############################################################################
###############################################################################

def main():
    # Init logger6
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path,
                            'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(
        sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(
        torch.backends.cudnn.version()), log)


    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log')
    logger = Logger(tb_path)
    writer = SummaryWriter(tb_path)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'mnist':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    if args.dataset == 'mnist':
        train_data = dset.MNIST(
            args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.MNIST(args.data_path, train=False,
                               transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(
            args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(
            args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(
            args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(
            args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path, split='train',
                               transform=train_transform, download=True)
        test_data = dset.SVHN(args.data_path, split='test',
                              transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(
            args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.STL10(args.data_path, split='test',
                               transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)

    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes, col_size=args.col_size, group_size=args.group_size, ADCprecision=args.ADCprecision, cellBit=args.cell_bit)
    print_log("=> network :\n {}".format(net), log)

    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    for name, value in net.named_parameters():
        print(name)
    # optionally resume from a checkpoint
    if args.resume:
        new_state_dict = OrderedDict()
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            if not (args.fine_tune):
                args.start_epoch = checkpoint['epoch']
                recorder = checkpoint['recorder']

            state_tmp = net.state_dict()
            for k, v in checkpoint['state_dict'].items():
                print(k)
                name = k
                # name = k[7:]
                print(name)

                new_state_dict[name] = v

            if 'state_dict' in checkpoint.keys():
                state_tmp.update(new_state_dict)
            else:
                print('loading from pth file not tar file')
                state_tmp.update(new_state_dict)

            net.load_state_dict(state_tmp)


            print_log("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, args.start_epoch), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        # summary(net, (3,32,32))
        validate(test_loader, net, criterion, log)
        return

def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    if args.adc_infer:
        alpha = []
        for name, param in model.named_parameters():
            if 'alpha' in name:
                print(f'alpha:{param.item()} | name: {name}')
                alpha.append(param.item())
        count = 0
        # ========for ResNet ============#
        # for m in model.modules():
        #     if isinstance(m, Qconv2d):
        #         if count == 0:
        #             m.act_alpha = alpha[-1]
        #             m.layer_idx = count
        #         else:
        #             m.layer_idx = count
        #             m.act_alpha = alpha[count-1]
        #         count += 1

        #     if isinstance(m, QLinear):
        #         m.act_alpha = alpha[count-1]

        # ========for VGG ============#
        for m in model.modules():
            if isinstance(m, Qconv2d):
                m.layer_idx = count
                m.act_alpha = alpha[count]
                count += 1

            if isinstance(m, QLinear):
                m.act_alpha = alpha[count]
                count += 1

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda()
                input = input.cuda()

            for m in model.modules():
                if isinstance(m, Qconv2d):
                    m.iter = i
            
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            
            # print(i)
            # if i == 0:
            #     break
        print_log(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                 error1=100 - top1.avg),log)

    return top1.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
