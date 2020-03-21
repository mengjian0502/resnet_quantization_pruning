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
import csv

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import models
from models.ResNet_tern_cifar import filterternConv2d, pruneLinear, quanConv2d, _quanFunc
from models.ResNet_cifar_vanilla import filterpruneConv2d, pLinear
from logger import Logger
# import yellowFin tuner
# sys.path.append("./tuner_utils")
# from tuner_utils.yellowfin import YFOptimizer

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
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--optimizer', type=str, default='SGD',
                    choices=['SGD', 'Adam', 'YF'])
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float,
                    default=0.001, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=1e-4,
                    help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 200)')
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
parser.add_argument('--save_freq', default=50, type=int, help='checkpoint saving frequency')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='device range [0,ngpu-1]')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default=5000, help='manual seed')

# activation clipping
parser.add_argument('--clp', dest='clp',
                    action='store_true', help='using clipped relu in each stage')
parser.add_argument('--a_lambda', type=float,
                    default=0.01, help='The parameter of alpha L2 regularization')
# weight clipping
parser.add_argument('--w_clp',action='store_true', help='using clipped quantized level')
parser.add_argument('--b_lambda',type=float,
                    default=0.01, help='The parameter of beta L2 regularization')

# group lasso
parser.add_argument('--swp', dest='swp',
                    action='store_true', help='using structured pruning')
parser.add_argument('--lamda', type=float,
                    default=0.001, help='The parameter for swp.')
parser.add_argument('--ratio', type=float,
                    default=0.25, help='pruned ratio')
parser.add_argument('--group_ch', type=int,
                    default=16, help='group size for group lasso')

parser.add_argument('--gradual_prune', dest='gradual_prune',
                    action='store_true', help='using structured pruning')

parser.add_argument('--skip_downsample', type=int, default=1, help='compress layer of model')

parser.add_argument('--layer_begin', type=int, default=1,  help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=1,  help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1,  help='compress layer of model')

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
    print_log("Weight Decay: {}".format(args.decay), log)
    print_log("Lamda: {}".format(args.lamda), log)
    print_log("Compress Ratio: {}".format(args.ratio), log)

  

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
            transforms.RandomCrop(32, padding=4),
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
    net = models.__dict__[args.arch](num_classes)
    print_log("=> network :\n {}".format(net), log)
    # inputsize = torch.tensor((3, 224, 224))
    # summary(net, inputsize)
    # net.load_state_dict(torch.load('save/mobilenetv2_1.pth'))
    #set first and last layer grad to false
    # if args.fine_tune:
    #     print('no grad for first and last layer')
    #     if args.dataset == 'imagenet':
    #         net.conv1.weight.requires_grad = False
    #         net.fc.weight.requires_grad = False
    #     if args.dataset == 'cifar10':
    #         net.conv_1_3x3.weight.requires_grad = False
    #         net.classifier.weight.requires_grad = False
    # inputsize = torch.tensor((1, 28, 28))
    # summary(net, inputsize)
    if args.use_cuda:
        if args.ngpu > 1:
            # net = torch.nn.DataParallel(net, device_ids=[1,2])
            net = torch.nn.DataParallel(net)

    # define loss function (criterion) and optimizer

    criterion = torch.nn.CrossEntropyLoss()

    if args.optimizer == "SGD":
        print("using SGD as optimizer")
        optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, net.parameters()),
                                    lr=state['learning_rate'],
                                    momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)

    elif args.optimizer == "Adam":
        print("using Adam as optimizer")
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, net.parameters()),
                                     lr=state['learning_rate'],
                                     weight_decay=state['decay'])

    elif args.optimizer == "YF":
        print("using YellowFin as optimizer")
        optimizer = YFOptimizer(filter(lambda param: param.requires_grad, net.parameters()), lr=state['learning_rate'],
                                mu=state['momentum'], weight_decay=state['decay'])

    elif args.optimizer == "RMSprop":
        print("using RMSprop as optimizer")
        optimizer = torch.optim.RMSprop(filter(lambda param: param.requires_grad, net.parameters()),
                                        lr=state['learning_rate'], alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)  # count number of epoches
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
                optimizer.load_state_dict(checkpoint['optimizer'])

            state_tmp = net.state_dict()
            for k, v in checkpoint['state_dict'].items():
                print(k)
                name = k
                # name = k[7:]
                print(name)

                new_state_dict[name] = v

            if 'state_dict' in checkpoint.keys():
                #state_tmp.update(new_state_dict['state_dict'])

                state_tmp.update(new_state_dict)
            else:
                print('loading from pth file not tar file')
                state_tmp.update(new_state_dict)
                #state_tmp.update(checkpoint)

            net.load_state_dict(state_tmp)
            #net.load_state_dict(torch.load('save/mobilenetv2_1.pth'))

            # net.load_state_dict(checkpoint['state_dict'])

            print_log("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, args.start_epoch), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        validate(test_loader, net, criterion, log)
        return

    if args.gradual_prune:
        m=Mask(net)
        m.init_length()
        comp_rate = 1 - args.ratio
        print("-"*10+"one epoch begin"+"-"*10)
        print("the compression rate now is %f" % comp_rate)


    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    w_zero_idx = []
    thre_mean = []
   # group_ch = 16   # group number
    spar = []
    penal = []
    
    
    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate, current_momentum = adjust_learning_rate(
            optimizer, epoch, args.gammas, args.schedule)
        # Display simulation time
        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.5f}][M={:1.2f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate,
                                                                                   current_momentum)
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)


        # train for one epoch
        if args.clp:
            train_acc, train_los, train_alpha = train(
                train_loader, net, criterion, optimizer, epoch, log)
            print_log(f"Epoch: {epoch}, Alpha: {train_alpha}", log)
        else:
            train_acc, train_los = train(
                train_loader, net, criterion, optimizer, epoch, log)

  
        # evaluate on validation set
        val_acc, val_los = validate(test_loader, net, criterion, log)

        if args.gradual_prune and epoch % 10 == 0:
            print(epoch)
            w_zero_layer = []
            for n in net.modules():
                # if isinstance(n, filterternConv2d) or isinstance(n, pruneLinear):
                if isinstance(n, pLinear) or isinstance(n, filterpruneConv2d):
                    idx = n.get_weight_idx()
                    w_zero_layer.append(idx)
            w_zero_idx.append(w_zero_layer)

        
            val_acc, val_los = validate(test_loader, net, criterion, log)

        is_best = val_acc > recorder.max_accuracy(istrain=False)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        if args.model_only:
            checkpoint_state = {'state_dict': net.state_dict()}
        else:
            checkpoint_state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }

        if epoch % args.save_freq == 0:
            save_checkpoint(checkpoint_state, is_best,
                            args.save_path, f'checkpoint_{epoch}.pth.tar', log)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # save addition accuracy log for plotting
        # accuracy_logger(base_dir=args.save_path,
        #                epoch=epoch,
        #                train_accuracy=train_acc,
        #                test_accuracy=val_acc)
        # sparisty_logger(base_dir=args.save_path,
        #                epoch=epoch,
        #                data=spar_epoch)
        # thre_logger(base_dir=args.save_path,
        #                epoch=epoch,
        #                data=thre_epoch)
    log.close()

    filename = os.path.join(args.save_path, 'w_zero_idx')
    np.save(filename, w_zero_idx)

def glasso(var, dim=0):
    return var.pow(2).sum(dim=dim).pow(1/2).sum()


def glasso_thre(var, dim=0):
    a = var.pow(2).sum(dim=dim).pow(1/2)
    mean_a  = a.mean()

    a = torch.min(a, args.ratio*mean_a)

    return a.sum()

def glasso_rank(var, group_ch, dim=0):
    a = var.pow(2).sum(dim=dim).pow(1/2)

    a_sort, f = a.sort()

            
    index = torch.tensor(int((args.ratio) * group_ch) * int(a.size(0)/group_ch)).int()

    thre = a_sort[index-1]
    a = torch.min(a, thre)

    return a.sum()

def glasso_mix(var, dim=0):
    a = var.pow(2).sum(dim=dim).pow(1/2)
    mean_a  = a.mean()

    a_sort, f = a.sort()

            
    index = torch.tensor(int((0.35) * 16) * int(a.size(0)/16)).int()

    thre = a_sort[index-1]
    thre1 = torch.min(mean_a, thre)
    a = torch.min(a, thre1)

    return a.sum()

# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.use_cuda:
            # the copy will be asynchronous with respect to the host.
            target = target.cuda(non_blocking=True)
            input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # # ============ add group lasso ============#
        #group size = [group_ch,3,3]

        if args.swp:
            lamda = torch.tensor(args.lamda).cuda()
            reg_g1 = torch.tensor(0.).cuda()
            group_ch = args.group_ch
            # == channel-wise defined 
            # for m in model.modules():
            #     if isinstance(m, nn.Conv2d):
            #         w_l = m.weight
            #         w_l = w_l.view(w_l.size(0), -1)
            #         reg_g1 += glasso_thre(w_l, 1)
            #     if isinstance(m, nn.Linear):
            #         w_f = m.weight
            #         reg_g2 += glasso_thre(w_f, 0)

            # loss += lamda * (reg_g1+reg_g2) 

            # == group-wise defined
            count = 0
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    if not count in [0]:
                        w_l = m.weight
                        kw = m.weight.size(2)
                        num_group = w_l.size(0) * w_l.size(1) // group_ch
                        w_l = w_l.view(w_l.size(0), w_l.size(1) // group_ch, group_ch, kw, kw)
                        w_l = w_l.contiguous().view((num_group, group_ch * kw * kw))
                        reg_g1 += glasso_thre(w_l, group_ch, 1)

                        # reg_g1 += glasso_rank(w_l, 1)
                    count += 1
                # if isinstance(m, nn.Linear):
                #     w_f = m.weight
                #     reg_g2 += glasso_thre(w_f, 0)
            loss += lamda * (reg_g1)

        if args.clp:
            reg_alpha = torch.tensor(0.).cuda()
            a_lambda = torch.tensor(args.a_lambda).cuda()

            alpha = []
            for name, param in model.named_parameters():
                if 'alpha' in name:
                    alpha.append(param.item())
                    reg_alpha += param.item() ** 2
            loss += a_lambda * (reg_alpha)

        if args.w_clp:
            reg_beta = torch.tensor(0.).cuda()
            b_lambda = torch.tensor(args.b_lambda).cuda()

            beta = []
            for name, param in model.named_parameters():
                if 'beta' in name and args.w_clp:
                    beta.append(param.item())
                    reg_beta += param.item() ** 2
            loss += b_lambda * (reg_beta)
        
        # # ============ add group lasso ============#

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg),
        log)

    if args.w_clp:
        print(f"beta: {beta}")
    if args.clp:
        return top1.avg, losses.avg, alpha
    else:
        return top1.avg, losses.avg


def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            if args.use_cuda:
                target = target.cuda(non_blocking=True)
                input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

        print_log(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                 error1=100 - top1.avg),
            log)

    return top1.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


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


def thre_logger(base_dir, epoch, data):
    file_name = 'thre.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs data\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['data'] = data
    with open(file_path, 'a') as thre_log:
        thre_log.write(
            '{epoch}       {data}    \n'.format(**recorder))

def sparisty_logger(base_dir, epoch, data):
    file_name = 'sparsity.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs data\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['data'] = data
    with open(file_path, 'a') as data_log:
        data_log.write(
            '{epoch}       {data}    \n'.format(**recorder))

def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write(
            '{epoch}       {train}    {test}\n'.format(**recorder))


class Mask:
    def __init__(self,model):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        
    
    def get_codebook(self, weight_torch,compress_rate,length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()
    
        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)
        
        threshold = weight_sort[int (length * (1-compress_rate) )]
        weight_np [weight_np <= -threshold  ] = 1
        weight_np [weight_np >= threshold  ] = 1
        weight_np [weight_np !=1  ] = 0
        
        print("codebook done")
        return weight_np


    def get_filter_codebook(self, weight_torch,compress_rate,length):
        codebook = np.ones(length)
        if len( weight_torch.size())==4:
            filter_pruned_num = int(weight_torch.size()[0]*(1-compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0],-1)
            norm2 = torch.norm(weight_vec,2,1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
#            norm1_sort = np.sort(norm1_np)
#            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] *weight_torch.size()[2] *weight_torch.size()[3]
            for x in range(0,len(filter_index)):
                codebook[filter_index[x] *kernel_length : (filter_index[x]+1) *kernel_length] = 0

            print("filter codebook done")
        else:
            pass
        return codebook
    
    def convert2tensor(self,x):
        x = torch.FloatTensor(x)
        return x
    
    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size [index] = item.size()
        
        for index1 in self.model_size:
            for index2 in range(0,len(self.model_size[index1])):
                if index2 ==0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]
                    
    def init_rate(self, layer_rate):
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate [index] = 1
        for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
            self.compress_rate[key]= layer_rate
        #different setting for  different architecture
        if args.arch == 'resnet20' or 'tern_resnet20':
            last_index = 57
            skip_list = [54]
        elif args.arch == 'resnet32':
            last_index = 93
        elif args.arch == 'resnet56':
            last_index = 165
        elif args.arch == 'resnet110':
            last_index = 327
        
        self.mask_index =  [x for x in range (0,last_index,3)]
        if args.skip_downsample == 1:
            for x in skip_list:
                self.compress_rate[x] = 1
                self.mask_index.remove(x)
                print(self.mask_index)
        else:
            pass       
#        self.mask_index =  [x for x in range (0,330,3)]
        
    def init_mask(self,layer_rate):
        self.init_rate(layer_rate)
        for index, item in enumerate(self.model.parameters()):
            if(index in self.mask_index):
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],self.model_length[index] )
                self.mat[index] = self.convert2tensor(self.mat[index])
                if args.use_cuda:
                    self.mat[index] = self.mat[index].cuda()
        print("mask Ready")

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if(index in self.mask_index):
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        print("mask Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            #            if(index in self.mask_index):
            if index in [x for x in range(args.layer_begin, args.layer_end + 1, args.layer_inter)]:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print("layer: %d, number of nonzero weight is %d, zero is %d" % (
                    index, np.count_nonzero(b), len(b) - np.count_nonzero(b)))


if __name__ == '__main__':
    main()
