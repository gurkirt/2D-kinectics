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

import os
import json
import pandas as pd

baseDir = "/mnt/mars-delta/kinetics/"


def parse_kinetics_annotations(input_csv):
    """Returns a parsed DataFrame.
    arguments:
    ---------
    input_csv: str
        Path to CSV file containing the following columns:
          'YouTube Identifier,Start time,End time,Class label'

    returns:
    -------
    dataset: DataFrame
        Pandas with the following columns:
            'video-id', 'start-time', 'end-time', 'label-name'
    """

    df = pd.read_csv(input_csv)
    df.rename(columns={'youtube_id': 'video-id',
                       'time_start': 'start-time',
                       'time_end': 'end-time',
                       'label': 'label-name',
                       'is_cc': 'is-cc'}, inplace=True)
    return df


def getlabels():

    train_df = parse_kinetics_annotations(baseDir+'hfiles/kinetics_train.csv')
    actions = dict()
    count = 0
    with open(baseDir+'hfiles/classes.txt', 'w') as f:
        for label in train_df['label-name']:
            if label not in actions.keys() and label != 'test':
                count += 1
                actions[label] = count
                f.write(label+'\n')
    print(count)
    return actions

def getDB(database, df, subset):
    print('Gathering for ', subset)
    traincount = 0
    classes = getlabels()
    for i, row in df.iterrows():
        label = row['label-name']
        videid = row['video-id']
        imagesdir = baseDir + 'rgb-images/' + videid
        vidinfo = dict()
        vidinfo['label'] = label
        if subset == 'test':
            vidinfo['cls'] = 0
        else:
            vidinfo['cls'] = classes[label]
        vidinfo['subset'] = subset
        
        if os.path.isdir(imagesdir):
            imagelist = os.listdir(imagesdir)
            imagelist = [d for d in imagelist if d.endswith('.jpg')]
            if len(imagelist) > 8:
                vidinfo['isthere'] = True
                vidinfo['numf'] = len(imagelist)
                traincount += 1
            else:
                vidinfo['isthere'] = False
        else:
            vidinfo['isthere'] = False
        database[videid] = vidinfo
    return database, traincount


def main():
    train_df = parse_kinetics_annotations(baseDir+'hfiles/kinetics_train.csv')
    val_df = parse_kinetics_annotations(baseDir+'hfiles/kinetics_val.csv')
    test_df = parse_kinetics_annotations(baseDir+'hfiles/kinetics_test.csv')
    classes = getlabels()
    database = dict()
    
    database, traincount = getDB(database, train_df, 'train')
    database, valcount = getDB(database, val_df, 'val')
    database, testcount = getDB(database, test_df, 'test')
    
    annot = dict()
    annot['database'] = database
    annot['classes'] = classes
    print('valcount', valcount, 'traincount', traincount, 'testcount', testcount)
    with open(baseDir+'hfiles/finalannots.json','w') as f:
        json.dump(annot, f)

if __name__ == '__main__':
    main()
