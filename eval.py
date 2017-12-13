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

from Evaluation.eval_kinetics import ANETclassification

def getscore(ground_truth_filename, prediction_filename, subset='val', verbose=True):
    anet_classification = ANETclassification(ground_truth_filename,
                                             prediction_filename,
                                             subset=subset, verbose=verbose,
                                             check_status=True, top_k=1)
    anet_classification.evaluate()

    anet_classification = ANETclassification(ground_truth_filename,
                                             prediction_filename,
                                             subset=subset, verbose=verbose,
                                             check_status=True, top_k=5)
    anet_classification.evaluate()




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
parser.add_argument('--test-iteration', default=500000, type=int, metavar='N',
                    help='manual iterations number (useful on restarts)')

## parameter for optimizer
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-tb', '--test-batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--ngpu', default=2, type=int, metavar='N',
                    help='use multiple GPUs take ngpu the avaiable GPUs')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--global-models-dir', default='~/global-models/pytorch-imagenet',
                    type = str, metavar='PATH', help='place where pre-trained models are ')

parser.add_argument('--pretrained', default=False, type=bool,
                    help='use pre-trained model default (True)')

## directory
parser.add_argument('--root', default='/mnt/mars-delta/',
                    type = str, metavar='PATH', help='place where datasets are present')

args = parser.parse_args()

gtfile = '/nfs01/REGdata-SSD02/gurkirt/kinetics/hfiles/finalannots.json'

exp_name = '{}-{}-{}-sl{:02d}-g{:d}-fs{:d}-{}-{:06d}'.format(args.dataset,
                args.arch, args.input, args.seq_len, args.gap, args.frame_step, args.batch_size, int(args.lr * 1000000))

args.exp_name = exp_name
args.root += args.dataset+'/'
model_save_dir = args.root + 'cache/' + exp_name

save_filename = '{:s}/output_{:s}_{:06d}.pkl'.format(model_save_dir, 'val', args.test_iteration)

for classtopk in [10, 30, 50]:
    outfilename = '{:s}-clstk-{:03d}.json'.format(save_filename[:-4], classtopk)
    print(outfilename)
    getscore(gtfile, outfilename)