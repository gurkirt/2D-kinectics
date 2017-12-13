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

import numpy as np
import shutil,os,json
from random import shuffle, randint
import scipy.io as sio

basedir = '/mnt/sun-alpha/kinetics/'
split = 'train'


def writevidlist(vlist, name):
    with open('{}hfiles/vid{}list-{}.txt'.format(basedir, name, split), 'w') as f:
        shuffle(vlist)
        for v in vlist:
            f.write(v)


def saveVidTrainTestList():

    with open(basedir+'hfiles/finalannots.json','r') as f:
        vidinfo = json.load(f)


    database = vidinfo["database"]
    print('number of videos downloded are ', len(database.keys()))
    trainlist = []
    validlist = []
    testlist = []
    count = 0

    for vid, videoname in enumerate(database.keys()):
        vidinfo = database[videoname]
        isthere = vidinfo['isthere']
        if isthere and vidinfo['subset'] != 'test':
            istrain = vidinfo['subset'] == 'train' or split != 'train'
            numf = vidinfo['numf']
            label = vidinfo['cls']
            if istrain:
                trainlist.append('{}:{}:{}\n'.format(videoname, str(numf), str(label)))
                count += 1
            else:
                validlist.append('{}:{}:{}\n'.format(videoname, str(numf), str(label)))
        elif isthere:
            numf = vidinfo['numf']
            testlist.append('{}:{}:{}\n'.format(videoname, str(numf), str(1)))
    if split == 'train':
        writevidlist(validlist, 'Valid')

    writevidlist(trainlist, 'Train')
    writevidlist(validlist, 'Val')
    writevidlist(testlist, 'Test')

    print('Number of trainvids ',len(trainlist), 'validation ',len(validlist),' testing ',len(testlist))

if __name__ == '__main__':
    saveVidTrainTestList()
