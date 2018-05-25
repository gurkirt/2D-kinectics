'''

Author: Gurkirt Singh
Start data: 2nd May 2016
purpose: of this file is to take all .mp4 videos and convert them to jpg images

'''

import numpy as np
import math,pickle,shutil,os
import json
import string
import pandas as pd

baseDir_src = "/mnt/sun-gamma/backup-videos/kinetics/"
baseDir_dst = "/mnt/mars-delta/kinetics/"

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


def getlabels(kin_subset):

    train_df = parse_kinetics_annotations(baseDir_dst+'hfiles/kinetics_{}_train.csv'.format(kin_subset))
    actions = dict()
    count = 0
    with open(baseDir_dst+'hfiles/classes_{}.txt'.format(kin_subset), 'w') as f:
        for label in train_df['label-name']:
            if label not in actions.keys() and label != 'test':
                count += 1
                actions[label] = count
                f.write(label+'\n')
    print(count)
    return actions


def getDB(database, classes, df, subset, kin_subset):
    print('Gathering for ', subset)
    traincount = 0
    # classes = getlabels(kin_subset)
    for i, row in df.iterrows():
        label = row['label-name']
        videid = row['video-id']
        imagesdir = baseDir_dst + 'rgb-images/' + videid
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
            if len(imagelist) > 15:
                vidinfo['isthere'] = True
                vidinfo['numf'] = len(imagelist)
                traincount += 1
            else:
                vidinfo['isthere'] = False
        else:
            vidinfo['isthere'] = False
        database[videid] = vidinfo

    return database, traincount


def convanem(name):
    newname = []
    for c in name:
        if c in [' ', '(', ')',"'"]:
            newname.append('\\' + c)
        else:
            newname.append(c)
    return ''.join(newname)


def extract_frames(df, fps=16, trim_format='%06d'):
    count = 0
    for vid, row in df.iterrows():

        vidfile = baseDir_src + 'videos/' + row['label-name']+ '/%s_%s_%s.mp4' % (row['video-id'], trim_format % row['start-time'], trim_format % row['end-time'])
        imgdir = baseDir_dst + 'rgb-images/' + row['video-id'] + '/'

        # print('\n\n',count, vid, row['video-id'], vidfile, '\n\n',imgdir,'\n')
        if not os.path.isdir(imgdir):
            os.mkdir(imgdir)

        if os.path.isfile(vidfile):
            count += 1

        imglist = os.listdir(imgdir)
        imglist = [i for i in imglist if i.endswith('.jpg')]
        if len(imglist) < 0 and os.path.isfile(vidfile):
            vidfile = convanem(vidfile)
            if fps > 0:
                cmd = 'ffmpeg -i {} -qscale:v 15 -r {} {}%05d.jpg'.format(vidfile, fps, imgdir)  ##-vsync 0
            else:
                cmd = 'ffmpeg -i {} -qscale:v 15 {}%05d.jpg'.format(vidfile, imgdir)  # -vsync 0
            # PNG format is very storage heavy so I choose jpg.
            # images will be generated in JPG format with quality scale = 15; you can adjust according to you liking
            print(vid, cmd)
            # f.write(cmd+'\n') ## this could act a log file
            os.system(cmd)
    print('counts', len(df), count)


def main():
    for kin_subset in ['400', '600']:
        train_df = parse_kinetics_annotations(baseDir_dst+'hfiles/kinetics_{}_train.csv'.format(kin_subset))
        val_df = parse_kinetics_annotations(baseDir_dst+'hfiles/kinetics_{}_val.csv'.format(kin_subset))
        test_df = parse_kinetics_annotations(baseDir_dst+'hfiles/kinetics_{}_test.csv'.format(kin_subset))
        classes = getlabels(kin_subset)

        print('NUMBER OF CLASSES ARE ', len(classes))

        # extract_frames(test_df)
        # extract_frames(train_df)
        # extract_frames(val_df)

        database = dict()
        database, traincount = getDB(database, classes, train_df, 'train', kin_subset)
        database, valcount = getDB(database, classes, val_df, 'val', kin_subset)
        database, testcount = getDB(database, classes, test_df, 'test', kin_subset)

        annot = dict()
        annot['database'] = database
        annot['classes'] = classes
        print('valcount', valcount, 'traincount', traincount, 'testcount', testcount)
        print('out of val', len(val_df), 'train', len(train_df), 'test', len(test_df))
        with open(baseDir_dst+'hfiles/Annots_{}.json'.format(kin_subset),'w') as f:
            json.dump(annot, f)


def get_yid(filename):
    names = []
    for line in open(filename).readlines():
        line = line.rstrip('\n')
        line = line.rstrip('\r')
        if len(line)>1:
            names.append(line)
        else:
            print(line)
    return names


def save_200_annots():

    with open(baseDir_dst + 'hfiles/Annots_{}.json'.format('400'), 'r') as f:
        annot = json.load(f)
    kin_subset = '200'
    train_yids = get_yid('{}hfiles/kinetics_{}_train.txt'.format(baseDir_dst, kin_subset))
    val_yids = get_yid('{}hfiles/kinetics_{}_val.txt'.format(baseDir_dst, kin_subset))
    ids = {}
    ids['train']  = train_yids
    ids['val'] = val_yids
    class_4k = annot['classes']
    db = annot['database']
    classes = dict()
    count = 0
    new_db = dict()

    for vid, videoname in enumerate(sorted(db.keys())):
        video_info = db[videoname]
        isthere = video_info['isthere']
        class_name = video_info['label']
        if isthere and video_info['subset'] in 'val' and class_name not in classes.keys() and videoname in ids['val']:
                classes[class_name] = count
                count += 1
    print('Number of classes', len(classes.keys()))
    for subset in ids.keys():
        print('Subset ', subset)
        for videoname in ids[subset]:
            video_info = db[videoname]
            if video_info['isthere']:
                # print(videoname)
                info = {}
                class_name = video_info['label']
                info['label'] = class_name
                info['cls'] = classes[class_name]
                info['isthere'] = video_info['isthere']
                info['numf'] = video_info['numf']
                info['subset'] = subset
                new_db[videoname] = info

    annot = dict()
    annot['database'] = new_db
    annot['classes'] = classes
    print('total videos with annotations in 200 case', len(new_db.keys()), len(classes.keys()))
    with open(baseDir_dst + 'hfiles/Annots_{}.json'.format(kin_subset), 'w') as f:
        json.dump(annot, f)

if __name__ == '__main__':
    # main()
    save_200_annots()