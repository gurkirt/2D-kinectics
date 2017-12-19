"""
Author: Gurkirt Singh
 https://github.com/Gurkirt

 Copyright (c) 2017, Gurkirt Singh

    This code and is available
    under the terms of MIT License provided in LICENSE.
    Please retain this notice and LICENSE if you use
    this file (or any portion of it) in your project.
    ---------------------------------------------------------

purpose: of this file is to define Kinetics dataset class so it can be used with
torch.util.dataloader class

"""

import os, torch, pdb
import numpy as np
import json
from PIL import Image
from PIL import ImageFile
import torch.utils.data as data
import random, cv2
import collections
from numpy import random as nprandom
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pilresize(img, size, interpolation=Image.BILINEAR):

    """Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """

    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * float(h) / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * float(w) / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def pil_random_crop(img, scale_size, output_size, params=None):
    img = pilresize(img, scale_size)
    th = output_size
    tw = output_size
    if params is None:
        w, h = img.size
        if w == tw and h == th:
            return img
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        flip = random.random()<0.5
    else:
        i,j,flip = params
    img = img.crop((j, i, j + tw, i + th))
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img, [i, j, flip]

def cv_random_crop(img, scale_size, output_size, params=None):

    if params is None:
        height, width, _ = img.shape
        w = nprandom.uniform(0.6 * width, width)
        h = nprandom.uniform(0.6 * height, height)
        left = nprandom.uniform(width - w)
        top = nprandom.uniform(height - h)
        # convert to integer rect x1,y1,x2,y2
        rect = np.array([int(left), int(top), int(left + w), int(top + h)])
        flip = random.random()<0.5
    else:
        rect,flip = params

    img = img[rect[1]:rect[3], rect[0]:rect[2], :]

    return img, [rect, flip]

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)

def cv_loader(path):
    #print('Going to use cv loader')
    return  cv2.imread(path)

def make_lists(annot_file, subsets, frame_step, seq_len=1, gap=1):
    with open(annot_file, 'r') as f:
        annoData = json.load(f)
    database = annoData["database"]
    classes = annoData["classes"]
    video_list = []
    video_labels = []
    vcount = -1
    image_list = []
    totalcount = 0
    for vid,videoname in enumerate(sorted(database.keys())):
        video_info = database[videoname]
        isthere = video_info['isthere']
        if isthere and video_info['subset'] in subsets:
            video_list.append(videoname)
            label = 0
            vcount += 1
            numf = video_info['numf']
            if numf > seq_len * 2:
                if 'test' not in subsets:
                    label = video_info['cls']-1
                maxf = numf-(seq_len//2)*gap-1
                indexs = np.arange((seq_len//2)*gap, maxf, frame_step)
                if indexs.shape[0] > 0:
                    for fid in indexs:
                        totalcount += 1
                        image_list.append([vcount, int(fid + 1), label])
            video_labels.append(label)
    print('{} Images loaded from {} videso'.format(totalcount, vcount))
    return image_list, video_list, classes, video_labels


class KINETICS(data.Dataset):
    """Kinetics
    input is image, target is annotation
    Arguments:
        root (string): path base dirctory of kinectics dataset
        input_type (string): input tuep for example rgb, farneback, brox etc

    """

    def __init__(self, root, input_type, transform=None, target_transform=None,
                 dataset_name='actnet', subsets=['train'], exp_name='',
                 netname='inceptionv3', scale_size=321, input_size=299,
                 frame_step=6, seq_len=1, gap=1):

        assert seq_len%2==1, 'seq len can only be a odd integer'
        self.root = root
        self.mode = 'train' in subsets
        self.scale_size = scale_size
        self.input_size = input_size
        self.exp_name = exp_name
        self.seq_len = seq_len
        self.gap = gap

        self.input_type = input_type
        self.subsets = subsets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        self.loader = pil_loader
        self.random_crop = pil_random_crop
        if netname.find('vgg')>-1:
            self.loader = cv_loader
            self.random_crop = cv_random_crop

        self.annot_file = self.root + "hfiles/finalannots.json"
        print('Annot File: ', self.annot_file, ' Mode is set to ', self.mode)

        self.img_path = os.path.join('/mnt/mars-fast/datasets/kinetics/', input_type+'-images', '%s.jpg')
        #self.img_path = os.path.join(root, input_type + '-images', '%s.jpg')

        image_list, video_list, classes, video_labels = make_lists(self.annot_file, subsets, frame_step, seq_len=self.seq_len,gap=self.gap)

        #self.video_labels = video_labels
        self.classes = classes
        self.num_classes = len(classes.keys())
        self.video_list = video_list
        self.image_list = image_list
        print('Inistliased Kinetics date for ', subsets,' set with ', len(image_list),' images')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        imids = self.image_list[index]
        vid_num = int(imids[0])
        videoname = self.video_list[vid_num]
        frame_num = int(imids[1])
        target = np.int64(imids[2])
        half_len = self.seq_len//2
        gap = self.gap
        frame_nums = np.arange(frame_num-half_len*gap,frame_num+half_len*gap+1,gap)
        #print(frame_nums)
        assert len(frame_nums) == self.seq_len, ' frame indexs length should be the same as frame_nums'
        # if self.mode != 'test':
        imgs = []
        for fn in frame_nums:
            path = self.img_path % '{:s}/{:05d}'.format(videoname, fn)
            imgs.append(self.loader(path))
        # pdb.set_trace()
        #input_imgs = torch.FloatTensor(self.seq_len*3,input_size,input_size)
        params = None
        if self.transform is not None:
            for ind in range(self.seq_len):
                if self.mode:
                    imgs[ind], params = self.random_crop(imgs[ind], self.scale_size, self.input_size, params=params)
                imgs[ind] = self.transform(imgs[ind])
                imgs[ind] = imgs[ind].squeeze()
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.seq_len == 1:
            input_imgs = imgs[0]
        else:
            input_imgs = torch.cat(imgs, 0)

        #print('Done Stacking', input_imgs.size())
        # else:
        #     path = self.img_path % '{:s}/{:05d}'.format(videoname, frame_num)
        #     input = self.transform(self.loader(path))
        # print('single image dim', input.size())

        return input_imgs, target, vid_num, frame_num

    def __len__(self):
        return len(self.image_list)
