"""
    Copyright (c) 2017, Gurkirt Singh

    This code and is available
    under the terms of MIT License provided in LICENSE.
    Please retain this notice and LICENSE if you use
    this file (or any portion of it) in your project.
    ---------------------------------------------------------

    Purpose of this script is to creat VGG16 network and define its forward pass

"""

import torch, pdb
import torch.nn as nn

# This function is copied from
# https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py
def vggconv(in_channels, batch_norm=False):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers


def vggnet(pretrained=False, num_classes = 1000, global_models_dir = '', num_channels=3):

    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if pretrained:
        model = VGG16(vggconv(num_channels), num_classes)
        conv_path = global_models_dir + '/vgg16_reducedfc.pth'
        # model_path = global_models_dir + '/inception_v3_kinetics_tsn.pth'
        # model_path = global_models_dir + '/inception_v3_pretrained_actnet_cls.pth'
        print('=> From: ', conv_path)
        print('MODEL TYPE is STD')
        conv_dict = torch.load(conv_path)
        model.conv_base.load_state_dict(conv_dict)
        return model
    else:
        return VGG16(vggconv(), num_classes)


class VGG16(nn.Module):

    def __init__(self, convs, num_classes):
        super(VGG16, self).__init__()
        self.num_classes = num_classes
        self.conv_base = nn.ModuleList(convs)
        self.extra_layers = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                           nn.Conv2d(1024, 1024, kernel_size=3, stride=3), nn.ReLU(inplace=True)])
        ms = 3
        self.classifier = nn.Sequential(nn.Dropout(0.6),
                                        nn.Linear(1024*ms*ms, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(0.6),
                                        nn.Linear(4096, num_classes)
                                        )

    def forward(self, x):
        for k in range(len(self.conv_base)):
            x = self.conv_base[k](x)
        #print('xsize', x.size()) # 512x19x19
        #pdb.set_trace()
        for k in range(len(self.extra_layers)):
            x = self.extra_layers[k](x)
        #print('xsize', x.size()) #1024x3x3
        x = x.view(x.size(0),-1)
        #print('xsize', x.size())
        return self.classifier(x)

