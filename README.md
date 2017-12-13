# Kinectics training on 1 GPU in 2Days
This is [PyTorch](http://pytorch.org/) implementation of two stream network of action classification on [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) dataset.
We train two streams of networks independently on individual frames of RGB (appearence) and optical flow (flow) as inputs.

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
- We also support [Visdom](https://github.com/facebookresearch/visdom) for visualization of loss and accuracy on subset of validation set during training!
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

- Notes:
 * Use latest youtube-dl
 * Some video might be missing but you should be alright, if are able to download around 290K videos.

##### Preprocess
First we need to extra images out of videos using `ffmpeg` and resave the annotations.
You can take help of scripts in `prep` folder in the repo.
You can comute optical flow images using [optical-flow](https://github.com/gurkirt/optical-flow).
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
--visdom=True --input_type=rgb
```

To train of flow inputs
```Shell
CUDA_VISIBLE_DEVICES=1 python train.py --root=/home/user/kinetics/ global_models_dir=/home/user/pretrained-models/
--visdom=True --input_type=farneback --stepvalues=300000,425000
```

Different paramneter in `train.py` will result in different performance

- Note:
  * InceptionV3 occupies almost 8.5GB VRAM on a GPU, 
   raining can take from 3-7 days  depending upon the disk, cpu and gpu speed. 
   I used 1080Ti gpu, SSD hard-drive and an i7 cpu.  
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
  as well as compute frame-level `top1 & top3` accuracies 500K iteration.
  * There is a log file file created for frame-level evaluation.

##### Video-level evaluation
You will need frame-level scores

## Performance

<table style="width:100% th">
  <tr>
    <td> method </td>
    <td>f-top1</td> 
    <td>f-top3</td>
    <td>v-top1</td>
    <td>v-top5</td>
    <td>mean-error</td>
  </tr>
  <tr>
    <td align="left">RGB</td> 
    <td> soon </td>
    <td> soon </td>
    <td> soon </td> 
    <td> soon </td>
    <td> soon </td>
  </tr>
  <tr>
    <td align="left">Flow</td> 
    <td> soon </td>
    <td> soon </td>
    <td> soon </td> 
    <td> soon </td>
    <td> soon </td>
  </tr>
  <tr>
    <td align="left">RGB+FLOW</td> 
    <td> soon </td>
    <td> soon </td>
    <td> soon </td> 
    <td> soon </td>
    <td> soon </td>
  </tr>
</table>

## Extras (comming soon)
To use pre-trained model download the pretrained weights from the links given below and make changes in `test.py` to accept the downloaded weights. 

##### Download pre-trained networks
- Currently, we provide the following PyTorch models: 
    * InceptionV3 trained on kinectics ; available from my [google drive]()
      - appearence model trained on rgb-images (named `r`)
      - accurate flow model trained on farneback-images (named `f`)    

## TODO
 - Upload pretrained models
 - fill the table

## References
- [1] Kay, Will, et al. "The Kinetics Human Action Video Dataset." arXiv preprint arXiv:1705.06950 (2017).
