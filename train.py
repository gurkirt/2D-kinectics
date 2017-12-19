"""
Author: Gurkirt Singh
 https://github.com/Gurkirt

    Copyright (c) 2017, Gurkirt Singh

    This code and is available
    under the terms of MIT License provided in LICENSE.
    Please retain this notice and LICENSE if you use
    this file (or any portion of it) in your project.
    ---------------------------------------------------------
"""

import argparse
import os, socket, pdb
import time
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from models.model_init import initialise_model
from data.kinetics import KINETICS
from torch.optim.lr_scheduler import MultiStepLR
from data import BaseTransform
from utils import  accuracy, AverageMeter, save_checkpoint, get_mean_size

np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', metavar='NAME', default='kinetics',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='inceptionV3',
                    help='model architectures ')

## parameters for dataloader
parser.add_argument('--input', '-i', metavar='INPUT', default='rgb',
                    help='input image type')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--seq_len', default=1, type=int, metavar='N',
                    help='seqence length')
parser.add_argument('--gap', default=1, type=int, metavar='N',
                    help='gap between the input frame within a sequence')
parser.add_argument('--frame_step', default=6, type=int, metavar='N',
                    help='sample every frame_step for for training')
parser.add_argument('--max_iterations', default=400000, type=int, metavar='N',
                    help='number of total iterations to run')
parser.add_argument('--start-iteration', default=0, type=int, metavar='N',
                    help='manual iterations number (useful on restarts)')

## parameter for optimizer
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--ngpu', default=1, type=int, metavar='N',
                    help='use multiple GPUs take ngpu the avaiable GPUs')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--stepvalues', default='200000,300000', type=str,
                    help='Chnage the lr @')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')

## logging parameters
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=False, type=bool, metavar='B',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--visdom', default=False, type=bool, metavar='B',
                    help='weather to use visdom (default: True)')
parser.add_argument('--global_models_dir', default='~/global-models/pytorch-imagenet',
                    type = str, metavar='PATH', help='place where pre-trained models are ')
parser.add_argument('--pretrained', default=True, type=bool,
                    help='use pre-trained model default (True)')

## directory
parser.add_argument('--root', default='/mnt/mars-delta/',
                    type = str, metavar='PATH', help='place where datasets are present')

def main():
    val_step = 25000
    val_steps = [5000, ]
    train_step = 500

    args = parser.parse_args()
    hostname = socket.gethostname()

    args.stepvalues = [int(val) for val in args.stepvalues.split(',')]

    exp_name = '{}-{}-{}-sl{:02d}-g{:d}-fs{:d}-{}-{:06d}'.format(args.dataset,
                args.arch, args.input, args.seq_len, args.gap, args.frame_step, args.batch_size, int(args.lr * 1000000))

    args.exp_name = exp_name
    args.root += args.dataset+'/'
    model_save_dir = args.root + 'cache/' + exp_name
    if not os.path.isdir(model_save_dir):
        os.system('mkdir -p ' + model_save_dir)

    args.model_save_dir = model_save_dir
    args.global_models_dir = os.path.expanduser(args.global_models_dir)

    if args.visdom:
        import visdom
        viz = visdom.Visdom()
        ports = {'mars':8097,'sun':8096}
        viz.port = ports[hostname]
        viz.env = exp_name

        # initialize visdom loss plot
        loss_plot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 2)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Losses',
                title='Train & Val Losses',
                legend=['Train-Loss', 'Val-Loss']
            )
        )

        eval_plot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 4)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Accuracy',
                title='Train & Val Accuracies',
                legend=['trainTop3', 'valTop3', 'trainTop1','valTop1']
            )
        )

    ## load dataloading configs
    input_size, means, stds = get_mean_size(args.arch)
    normalize = transforms.Normalize(mean=means,
                                     std=stds)

    # Data loading transform based on model type
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    val_transform = transforms.Compose([transforms.Scale(int(input_size * 1.1)),
                                        transforms.CenterCrop(int(input_size)),
                                        transforms.ToTensor(),
                                        normalize,
                                        ])

    if args.arch.find('vgg') > -1:
        transform = BaseTransform(size=input_size,mean=means)
        val_transform = transform
        print('\n\ntransforms are going to be VGG type\n\n')

    train_dataset = KINETICS(args.root,
                             args.input,
                             transform,
                             netname=args.arch,
                             subsets=['train'],
                             scale_size=int(input_size*1.1),
                             input_size=int(input_size),
                             exp_name=exp_name,
                             frame_step=args.frame_step,
                             seq_len=args.seq_len,
                             gap=args.gap
                             )

    args.num_classes = train_dataset.num_classes
    print('Models will be cached in ', args.model_save_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True
                                               )

    val_dataset = KINETICS(args.root,
                           args.input,
                           val_transform,
                           netname=args.arch,
                           subsets=['val'],
                           exp_name=exp_name,
                           scale_size=int(input_size * 1.1),
                           input_size=int(input_size),
                           frame_step=args.frame_step*6,
                           seq_len=args.seq_len,
                           gap=args.gap
                           )

    val_loader  = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True
                                              )

    model, criterion = initialise_model(args)

    parameter_dict = dict(model.named_parameters())
    params = []
    for name, param in parameter_dict.items():
        if name.find('bias') > -1:
            params += [{'params': [param], 'lr': args.lr*2, 'weight_decay': 0}]
        else:
            params += [{'params': [param], 'lr': args.lr, 'weight_decay':args.weight_decay}]


    optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum)
    scheduler = MultiStepLR(optimizer, milestones=args.stepvalues, gamma=args.gamma)

    if args.resume:
        latest_file_name = '{:s}/latest.pth'.format(args.model_save_dir)
        print('Resume model from ', latest_file_name)
        latest_dict = torch.load(latest_file_name)
        args.start_iteration = latest_dict['iteration'] + 1
        model.load_state_dict(torch.load(latest_dict['model_file_name']))
        optimizer.load_state_dict(torch.load(latest_dict['optimizer_file_name']))
        log_fid = open(args.model_save_dir + '/training.log', 'a')
        args.iteration = latest_dict['iteration']-1
        for _ in range(args.iteration):
            scheduler.step()
    else:
        log_fid = open(args.model_save_dir + '/training.log', 'w')

    log_fid.write(args.exp_name + '\n')
    for arg in vars(args):
        print(arg, getattr(args, arg))
        log_fid.write(str(arg) + ': ' + str(getattr(args, arg)) + '\n')
    log_fid.write(str(model))
    best_top1 = 0.0
    val_loss = 0.0
    val_top1 = 0.0
    val_top3 = 0.0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    iteration = args.start_iteration
    approx_epochs = np.ceil(float(args.max_iterations-iteration)/len(train_loader))

    print('Approx Epochs to RUN: {}, Start Ietration {} Max iterations {} # of samples in dataset {}'.format(
        approx_epochs, iteration, args.max_iterations, len(train_loader)))
    epoch = -1

    model.train()
    torch.cuda.synchronize()
    start = time.perf_counter()
    while iteration < args.max_iterations:
        epoch += 1
        for i, (batch, targets, __ , __) in enumerate(train_loader):
            if i<len(train_loader):
                if iteration > args.max_iterations:
                    break
                iteration += 1
                #pdb.set_trace()
                #print('input size ',batch.size())
                targets = targets.cuda(async=True)
                input_var = torch.autograd.Variable(batch.cuda(async=True))
                target_var = torch.autograd.Variable(targets)

                torch.cuda.synchronize()
                data_time.update(time.perf_counter() - start)

                # compute output
                output = model(input_var)
                loss = criterion(output, target_var)
                #pdb.set_trace()
                # measure accuracy and record loss
                prec1, prec3 = accuracy(output.data, targets, topk=(1, 3))
                losses.update(loss.data[0], batch.size(0))
                top1.update(prec1[0], batch.size(0))
                top3.update(prec3[0], batch.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                # measure elapsed time
                torch.cuda.synchronize()
                batch_time.update(time.perf_counter() - start)
                start = time.perf_counter()
                if iteration % args.print_freq == 0:
                    line = 'Epoch: [{0}][{1}/{2}] Time {batch_time.val:.3f} ({batch_time.avg:.3f}) Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                           epoch, iteration, len(train_loader), batch_time=batch_time, data_time=data_time)
                    line += 'Loss {loss.val:.4f} ({loss.avg:.4f}) Prec@1 {top1.val:.3f} ({top1.avg:.3f}) Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                                loss=losses, top1=top1, top3=top3)
                    print(line)
                    log_fid.write(line+'\n')

                avgtop1 = top1.avg
                avgtop3 = top3.avg
                avgloss = losses.avg
                if (iteration % val_step == 0 or iteration in val_steps) and iteration > 0:
                    # evaluate on validation set
                    val_top1, val_top3, val_loss = validate(args, val_loader, model, criterion)
                    line = '\n\nValidation @ {:d}: Top1 {:.2f} Top3 {:.2f} Loss {:.3f}\n\n'.format(
                                iteration, val_top1, val_top3, val_loss)
                    print(line)
                    log_fid.write(line)
                    # remember best prec@1 and save checkpoint
                    is_best = val_top1 > best_top1
                    best_top1 = max(val_top1, best_top1)
                    torch.cuda.synchronize()
                    line = '\nBest Top1 sofar {:.3f} current top1 {:.3f} Time taken for Validation {:0.3f}\n\n'.format(best_top1, val_top1, time.perf_counter()-start)
                    log_fid.write(line + '\n')
                    print(line)
                    save_checkpoint({
                        'epoch': epoch,
                        'iteration': iteration,
                        'arch': args.arch,
                        'val_top1': val_top1,
                        'val_top3': val_top3,
                        'val_loss': val_loss,
                        'train_top1': avgtop1,
                        'train_top3': avgtop3,
                        'train_loss': avgloss,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, args.model_save_dir)
                    if args.visdom:
                        viz.line(
                            X=torch.ones((1, 2)).cpu() * iteration,
                            Y=torch.Tensor([avgloss, val_loss]).unsqueeze(0).cpu(),
                            win=loss_plot,
                            update='append'
                        )
                        viz.line(
                            X=torch.ones((1, 4)).cpu() * iteration,
                            Y=torch.Tensor([avgtop3, val_top3, avgtop1, val_top1]).unsqueeze(0).cpu(),
                            win=eval_plot,
                            update='append'
                        )

                    model.train()

                if iteration % train_step == 0 and iteration > 0:
                    if args.visdom:
                        viz.line(
                            X=torch.ones((1, 2)).cpu() * iteration,
                            Y=torch.Tensor([avgloss, val_loss]).unsqueeze(0).cpu(),
                            win=loss_plot,
                            update='append'
                        )
                        viz.line(
                            X=torch.ones((1, 4)).cpu() * iteration,
                            Y=torch.Tensor([avgtop3, val_top3, avgtop1, val_top1]).unsqueeze(0).cpu(),
                            win=eval_plot,
                            update='append'
                        )
                    top1.reset()
                    top3.reset()
                    losses.reset()
                    print('RESET::=> ', args.exp_name)


def validate(args, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    torch.cuda.synchronize()
    end = time.perf_counter()

    for itr, (batch, targets,  __ , __) in enumerate(val_loader):
        if itr < len(val_loader):
            #print('input size ', batch.size())

            targets = targets.cuda(async=True)
            input_var = torch.autograd.Variable(batch.cuda(async=True), volatile=True)
            target_var = torch.autograd.Variable(targets, volatile=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            #pdb.set_trace()
            # measure accuracy and record lossargs.ex
            prec1, prec3 = accuracy(output.data, targets, topk=(1, 3))
            losses.update(loss.data[0], batch.size(0))
            top1.update(prec1[0], batch.size(0))
            top3.update(prec3[0], batch.size(0))

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            if itr % args.print_freq*10 == 0:
                print('Test:  [{0}/{1}] '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f}) '.format(
                       itr, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))

    return top1.avg, top3.avg, losses.avg


if __name__ == '__main__':
    main()