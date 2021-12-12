# Simple Baselines for Human Pose Estimation and Tracking

## Introduction
This is an paddle implementation of [*Simple Baselines for Human Pose Estimation and Tracking*](https://arxiv.org/abs/1804.06208). 
This work provides baseline methods that are surprisingly simple and effective, thus helpful for inspiring and evaluating new ideas for the field.
State-of-the-art results are achieved on challenging benchmarks.

## Main Results
### Results on MPII val
| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1|
|---|---|---|---|---|---|---|---|---|---|
| 256x256_pose_resnet_50_d256d256d256 | 96.351 | 95.329 | 88.989 | 83.176 | 88.420 | 83.960 | 79.594 | 88.532 | 33.911 |

### Note:
- Flip test is used.

## Environment
The code is developed using python 3.7 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 1 NVIDIA V100 GPU cards.

## Quick start
### Installation
1. Install paddle >= v2.1.2

1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
   
4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── pose_estimation
   ├── README.md
   └── requirements.txt
   ```
   
### Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. We have converted them into json format, you also need to download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW00SqrairNetmeVu4) or [GoogleDrive](https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

### Training on MPII

```
python pose_estimation/train.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml
```

### Valid on MPII using pretrained models

```
python pose_estimation/valid.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar
```

### Citation
If you use our code or models in your research, please cite with:
```
@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}
```
