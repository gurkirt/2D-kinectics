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

import argparse, json
import os, pdb, pickle
import time, socket
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from models.model_init import initialise_model
from data.kinetics import KINETICS
from utils import  accuracy, AverageMeter, get_mean_size

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

'''Define Save Directory'''

torch.manual_seed(0)
torch.cuda.manual_seed(0)

from Evaluation.eval_kinetics import ANETclassification
# from Evaluation.eval_classification import ANETclassification


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


def gettopklabel(preds, k, classtopk):
    scores = np.zeros(400)
    topk = min(classtopk, np.shape(preds)[1])
    for i in range(400):
        values = preds[i, :]
        values = np.sort(values)
        values = values[::-1]
        scores[i] = np.mean(values[:topk])
    sortedlabel = np.argsort(scores)[::-1]
    sortedscores = scores[sortedlabel]
    ss = sortedscores[:k]
    return sortedlabel[:k], ss/np.sum(ss[:5])


def getid2a(classes):
    actid2action = dict()
    for a in classes.keys():
        actid2action[str(classes[a])] = a
    return actid2action


def main():

    args = parser.parse_args()
    hostname = socket.gethostname()
    if hostname in ['mars', 'sun']:
        root = '/mnt/mars-delta/'
    else:
        raise 'PLEASE SPECIFY root FOR ' + hostname

    exp_name = '{}-{}-{}-sl{:02d}-g{:d}-fs{:d}-{}-{:06d}'.format(args.dataset,
                                                                 args.arch, args.input, args.seq_len, args.gap,
                                                                 args.frame_step, args.batch_size,
                                                                 int(args.lr * 1000000))

    args.exp_name = exp_name
    root += args.dataset + '/'
    args.root = root
    model_save_dir = root + 'cache/' + exp_name

    args.model_save_dir = model_save_dir
    args.global_models_dir = os.path.expanduser(args.global_models_dir)
    subset = 'val'

    args.subset = subset
    input_size, means, stds = get_mean_size(args.arch)
    print('means ', means)
    print('stds ', stds)

    normalize = transforms.Normalize(mean=means,std=stds)
    val_transform = transforms.Compose([transforms.Scale(int(input_size * 1.1)),
                                        transforms.CenterCrop(int(input_size)),
                                        transforms.ToTensor(),
                                        normalize,
                                        ])

    val_dataset = KINETICS(args.root,
                           args.input,
                           val_transform,
                           netname=args.arch,
                           subsets=['val'],
                           exp_name=exp_name,
                           scale_size=int(input_size * 1.1),
                           input_size=int(input_size),
                           frame_step=2,
                           seq_len=args.seq_len,
                           gap=args.gap
                           )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True
                                             )


    args.num_classes = val_dataset.num_classes

    save_filename = '{:s}/output_{:s}_{:06d}.pkl'.format(args.model_save_dir, subset, args.test_iteration)
    if not os.path.isfile(save_filename):
        print('Models will be cached in ', args.model_save_dir)
        log_fid = open(args.model_save_dir + '/test_log.txt', 'w')
        log_fid.write(args.exp_name + '\n')
        for arg in vars(args):
            print(arg, getattr(args, arg))
            log_fid.write(str(arg) + ': ' + str(getattr(args, arg)) + '\n')

        model, criterion = initialise_model(args)
        model_file_name = '{:s}/model_{:06d}.pth'.format(args.model_save_dir, args.test_iteration)
        print('Loading model from ', model_file_name)
        model_dict = torch.load(model_file_name)
        if args.ngpu>1:
            model.load_state_dict(model_dict)
        else:
            model.load_my_state_dict(model_dict)
        print('Done loading model')
        model.eval()
        log_fid.write(str(model))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()
        # switch to evaluate mode
        model.eval()
        end = time.time()
        allscores = dict()
        print('Starting to Iterate')
        for i, (batch, targets,  video_num, frame_nums) in enumerate(val_loader):
            targets = targets.cuda(async=True)
            input_var = torch.autograd.Variable(batch.cuda(async=True), volatile=True)
            target_var = torch.autograd.Variable(targets, volatile=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, targets, topk=(1, 3))
            losses.update(loss.data[0], batch.size(0))
            top1.update(prec1[0], batch.size(0))
            top3.update(prec3[0], batch.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test:   [{0}/{1}]'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top3=top3))
            res_data = output.data.cpu().numpy()
            # print('video_num type', video_num.type())
            # print('frame_num type', frame_nums.type())
            for k in range(res_data.shape[0]):
                videoname = val_dataset.video_list[int(video_num[k])]
                frame_num = int(frame_nums[k])
                if videoname not in allscores.keys():
                    allscores[videoname] = dict()
                    allscores[videoname]['scores'] = np.zeros((100, val_dataset.num_classes), dtype=np.float)
                    allscores[videoname]['fids'] = np.zeros(100, dtype=np.int16)
                    allscores[videoname]['count'] = 0

                scores = res_data[k, :]
                count = allscores[videoname]['count']
                allscores[videoname]['scores'][count, :] = scores
                allscores[videoname]['fids'][count] = frame_num
                allscores[videoname]['count'] += 1
        print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'.format(top1=top1, top3=top3))
        print('Done FRAME LEVEL evaluation Con')

        for videoname in allscores.keys():
            count = allscores[videoname]['count']
            allscores[videoname]['scores'] = allscores[videoname]['scores'][:count]
            fids = allscores[videoname]['fids'][:count]
            sortedfidsinds = np.argsort(fids)
            fids = fids[sortedfidsinds]
            allscores[videoname]['scores'] = allscores[videoname]['scores'][sortedfidsinds]
            allscores[videoname]['fids'] = fids

        with open(save_filename, 'wb') as f:
            pickle.dump(allscores,f)
    else:
        with open(save_filename, 'rb') as f:
            allscores = pickle.load(f)

    evaluate(allscores, val_dataset.annot_file, save_filename, subset)


def evaluate(allscores, annot_file, save_filename, subset):
    print(' ')
    with open(annot_file, 'r') as f:
        annotdata = json.load(f)
    database = annotdata["database"]
    classes = annotdata["classes"]
    print('smallest class ', min(classes.values()))
    actid2action = getid2a(classes)

    vdata = {}
    vdata['external_data'] = {'used': True, 'details': "inceptionNet V3 pretrained on imagenet dataset"}
    vdata['version'] = "KINETICS VERSION 1.0"

    K = 5
    for classtopk in [10,20,30,50]:
        outfilename = '{:s}-clstk-{:03d}.json'.format(save_filename[:-4], classtopk)
        print('outfile ', outfilename)
        print('Number of loaded', len(allscores.keys()))
        results = dict()
        nottherecount = 0
        for vid in database.keys():
            if database[vid]['subset'] == subset:
                vidresults = []
                if vid in allscores.keys():
                    preds = allscores[vid]['scores']
                    labels, scores = gettopklabel(np.transpose(preds), K, classtopk)
                    for idx in range(K):
                        score = scores[idx]
                        label = labels[idx]
                        name = actid2action[str(label+1)]
                        tempdict = {'label': name, 'score': score}
                        vidresults.append(tempdict)
                else:
                    vidresults = [{'label': actid2action[str(2)], 'score': 0.0000001}]
                    nottherecount += 1
                results[vid] = vidresults

        vdata['results'] = results
        with open(outfilename, 'w') as f:
            json.dump(vdata, f)
        print(annot_file,outfilename)
        # getscore(annot_file, outfilename)


if __name__ == '__main__':
    main()
