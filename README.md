# A Two Stream Baseline on Kinectics dataset
## Kinectics Training on 1 GPU in 2 Days
This is [PyTorch](http://pytorch.org/) implementation of two stream network of action classification on [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) dataset.
We train two streams of networks independently on individual(or stacked) frames of RGB (appearence) and optical flow (flow) as inputs.

Objective of this repository to establish a two stream baseline and ease the training process on
such a huge dataset.

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training'>Training</a>
- <a href='#building'>Evaluation</a>
- <a href='#todo'>TODO</a>
- <a href='#references'>Reference</a>

## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Install `ffmpeg`
- Please install cv2 as well for your python. I recommend using anaconda 3.6 and menpo's opnecv3 package.
- Clone this repository.
  * Note: We currently only support Python 3+ on Linux system
- We also support [Visdom](https://github.com/facebookresearch/visdom) 
for visualization of loss and accuracy on subset of validation set during training!
  * To use Visdom in the browser: 
  ```Shell
  # First install Python server and client 
  pip install visdom
  # Start the server (probably in a screen or tmux)
  python -m visdom.server --port=8097
  ```
  * Then (during training) navigate to http://localhost:8097/ (see the Training section below for more details).

## Dataset
Kinetics dataset can be Downloaded using [Crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).
<br>
Notes:
  * Use latest youtube-dl
  * Some video might be missing but you should be alright, if are able to download around 290K videos.

##### Preprocess
First we need to extract images out of videos using `ffmpeg` and resave the annotations,
so that annotations are compatible with this code.
<br>
You can take help of scripts in `prep` folder in the repo to do both the things.
<br>
You need to compute optical flow images using [optical-flow](https://github.com/gurkirt/optical-flow).
Compute `farneback` flow as it is much faster to compute and gives reasonable results. 
You might want to run multiple processes in parallel. 

## Training
- Download the pretrained weight for [InceptionV3](https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth)
 and [VGG-16](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth), 
 place them in same directory which will hold pertained models and set `global_models_dir` in `train.py`. 
- By default, we assume that you have downloaded that dataset.    
- To train the network of your choice simply specify the parameters listed in `train.py` as a flag or manually change them.

Let's assume that you extracted dataset in `/home/user/kinetics/` directory then your train command from the root directory of this repo is going to be:

```Shell
CUDA_VISIBLE_DEVICES=0 python train.py --root=/home/user/kinetics/ --global_models_dir=/home/user/pretrained-models/
--visdom=True --input_type=rgb --stepvalues=200000,350000 --max_iterations=500000
```

To train of flow inputs
```Shell
CUDA_VISIBLE_DEVICES=1 python train.py --root=/home/user/kinetics/ global_models_dir=/home/user/pretrained-models/
--visdom=True --input_type=farneback --stepvalues=250000,400000 --max_iterations=500000
```

Different paramneter in `train.py` will result in different performance

- Note:
  * InceptionV3 occupies almost 8.5GB VRAM on a GPU, 
   raining can take from 2-4 days  depending upon the disk, cpu and gpu speed.
   I used one 1080Ti gpu, SSD-PCIe hard-drive and an i7 cpu. Disk operation could be a bottleneck if you are using HDD.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section. 
    By default it is off.
  * If you don't like to use visdom then you always keep track of train using logfile which is saved under save_root directory
  * During training checkpoint is saved every 25K iteration also log it's frame-level `top1 & top3` 
    accuracies on a subset of 95k validation images.
  * We recommend to training for 500K iterations for all the input types.

## Evaluation
You can use `test.py` to generate frame-level scores and save video-level results in json file.
Further use `eval.py` to evaluate results on validation set 

##### produce frame-level scores
Once you have trained network then you can use `test.py` to generate frame-level scores.
Simply specify the parameters listed in `test.py` as a flag or manually change them. for e.g.:

```Shell
CUDA_VISIBLE_DEVICES=0 python3 test.py --root=/home/user/kinetics/ --input=rgb --test-iteration=500000
```
-Note
  * By default it will compute frame-level scores and store them 
  as well as compute frame-level `top1 & top3` accuracies using model from 500K-th iteration.
  * There is a log file file created for frame-level evaluation.

##### Video-level evaluation
Video-level labling requires frame-level scores.
`test.py` not only store frame-level score but also video-level scores in `evaluate`
function within. It will dump the video level output in json format
(same a used in activtiyNet challenge) for validation set.
Now you can specify the parameter in `eval.py` and evaluate

## Performance

<table style="width:100% th">
  <tr>
    <td> method </td>
    <td>Num Frames</td>
    <td>frame-top1</td>
    <td>frame-top3</td>
    <td>video-top1</td>
    <td>video-top5</td>
    <td>mean</td>
    <td>video-mAP</td>
  </tr>
  <tr>
    <td align="left">RGB</td>
    <td align="center">54.5</td>
    <td align="center">70.7</td>
    <td align="center">66.9</td>
    <td align="center">85.7</td>
    <td align="center">76.3</td>
    <td align="center">70.1</td>
  </tr>
  <tr>
    <td align="left">Flow</td> 
    <td align="center">25.7</td>
    <td align="center">39.7</td>
    <td align="center">45.1</td>
    <td align="center">69.7</td>
    <td align="center">57.4</td>
    <td align="center">46.6</td>
  </tr>
  <tr>
    <td align="left">RGB+FLOW</td> 
    <td align="center"> soon </td>
    <td align="center"> soon </td>
    <td align="center"> soon </td>
    <td align="center"> soon </td>
    <td align="center"> soon </td>
    <td align="center"> soon </td>
  </tr>
</table>

## Extras (comming soon)
Pre-trained models can be downloaded from the links given below.
You will need to make changes in `test.py` to accept the downloaded weights.

##### Download pre-trained networks
- Currently, we provide the following PyTorch models: 
    * InceptionV3 trained on kinectics ; available from my [google drive](https://drive.google.com/drive/folders/1ZzEMPepcGLEJ6dKIDqzsSpgCZX0pnZyw?usp=sharing)
      - appearence model trained on rgb-images (named `rgb_OneFrame_model_500000`)
      - accurate flow model trained on farneback-images (named `farneback_OneFrame_model_500000`)

## TODO
 - fill the table with fused results

## References
- [1] Kay, Will, et al. "The Kinetics Human Action Video Dataset." arXiv preprint arXiv:1705.06950 (2017).
