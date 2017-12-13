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

import torch
import shutil


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') > -1:
        print(classname, ', being put to eval mode')
        m.eval()

# model.apply(set_bn_eval)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, save_dir):
    itr = state['iteration']
    print('Save Dir: ', save_dir)
    model_file_name = '{:s}/model_{:06d}.pth'.format(save_dir, itr)
    optimizer_file_name = '{:s}/optimizer_{:06d}.pth'.format(save_dir, itr)
    print('Model File: ', model_file_name)
    print('Optim File: ', optimizer_file_name)
    torch.save(state['state_dict'], model_file_name)
    torch.save(state['optimizer'], optimizer_file_name)
    latest_file_name = '{:s}/latest.pth'.format(save_dir)
    latest_state = {
                'iteration': state['iteration'],
                'arch': state['arch'],
                'val_top1': state['val_top1'],
                'val_top3': state['val_top3'],
                'val_loss': state['val_loss'],
                'train_top1': state['train_top1'],
                'train_top3': state['train_top3'],
                'train_loss': state['train_loss'],
                'model_file_name': model_file_name,
                'optimizer_file_name' : optimizer_file_name
                    }
    torch.save(latest_state, latest_file_name)
    if is_best:
        best_file_name = '{:s}/best.pth'.format(save_dir)
        shutil.copyfile(latest_file_name, best_file_name)
    fid = open('{:s}/results_{:06d}.txt'.format(save_dir, itr),'w')
    fid.write('Val Top1 {:.3f}\nVal Top3 {:.3f}\nVal loss {:.3f}\nTrain Top1'
                        '{:.3f}\nTrain Top3 {:.3f}\nTrain loss {:.3f}'.format(
                        state['val_top1'], state['val_top3'], state['val_loss'],
                        state['train_top1'], state['train_top3'], state['train_loss']))
    fid.close()


def get_mean_size(arch):
    if arch.startswith('inceptionV3'):
        print('Selecting Mean STD and Size of inception network')
        input_size = 299.0
        means = [0.5, 0.5, 0.5]
        stds= [0.5, 0.5, 0.5]
    else:
        input_size = 300
        means = (104, 117, 123)
        stds = [0.0, 0.0, 0.0]
    return input_size, means, stds
