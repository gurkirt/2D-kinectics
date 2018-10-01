'''

Author: Gurkirt Singh
Start data: 2nd May 2017
purpose: of this file is to take all .mp4 videos and convert them to jpg images or mp3 files

'''

import os
import string
import pickle

baseDir = "/mnt/mercury-alpha1/moments/"

subset = 'validation'
subset = 'training'

def convanem(name):
    newname = []
    for c in name:
        if c in [' ', '(', ')', "'"]:
            newname.append('\\' + c)
        else:
            newname.append(c)
    return ''.join(newname)


def extractmp3(vids):
    count = 0
    audiolist = []
    for actdir, vidname in vids:
        count += 1
        vidfile = actdir + vidname
        vid = vidname[:11]
        mp3file = baseDir + 'audios/' + vid + '.mp3'
        vidfile = convanem(vidfile)
        cmd = 'ffmpeg -i {} -vn -acodec libmp3lame -ac 2 -ab 160k -ar 48000 {}'.format(vidfile, mp3file)
        os.system(cmd)
        audiolist.append(mp3file + '\n')
        print('\n\n\n', count, cmd, '\n\n')

    with open(baseDir + 'splitfiles/audiolist.txt', 'w') as f:
        for a in audiolist:
            f.write(a)


def extractframes(vids, fps):
    count = 0
    for actdir, vidname in vids:
        count += 1
        vidfile = actdir + vidname
        vid = vidname[:11]
        imgdir = '{}rgb-images/{}-{:012d}/'.format(baseDir, subset, count)
        print(count, vid, vidname)
        if not os.path.isdir(imgdir):
            os.mkdir(imgdir)
        #
        imglist = os.listdir(imgdir)
        imglist = [i for i in imglist if i.endswith('.jpg')]
        vidfile = convanem(vidfile)
        if len(imglist) < 10:
            if fps > 0:
                cmd = 'ffmpeg -i {} -qscale:v 15 -r {} {}%05d.jpg'.format(vidfile, fps, imgdir)  ##-vsync 0
            else:
                cmd = 'ffmpeg -i {} -qscale:v 15 {}%05d.jpg'.format(vidfile, imgdir)  # -vsync 0
            # PNG format is very storage heavy so I choose jpg.
            # images will be generated in JPG format with quality scale = 15; you can adjust according to you liking
            print(cmd)
            # f.write(cmd+'\n') ## this could act a log file
            os.system(cmd)
def extraction():
    for subset in ['validation', 'training']:
        allvideos = []
        downloaded = os.listdir(baseDir + 'videos/' + subset + '/')
        allcount = 0
        print('number of actions are', len(downloaded), downloaded)
        for act in sorted(downloaded):
            actdir = baseDir + 'videos/' + subset + '/' + act + '/'
            actvideos = os.listdir(actdir)
            print(act, 'has ', len(actvideos), 'vidoes')
            for d in sorted(actvideos):
                allvideos.append([actdir, d])
        with open(baseDir + subset + 'videoname_order.pkl', 'wb') as f:
            pickle.dump(allvideos, f)
        print('number of videos downloded are ', len(allvideos))
        ############################
        fps = 16  # set fps = 0 if you want to extract at original frame rate
        extractframes(allvideos, fps)


def save_db_info(allvideos, subset, db):
    mylist = []


if __name__ == '__main__':
    # extraction()
    db = dict()
    for subset in ['validation', 'training']:
        with open(baseDir + subset + 'videoname_order.pkl', 'rb') as f:
            allvideos = pickle.load(f)
        db = save_db_info(allvideos, subset, db)

