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
3. Install coco dependencies:
   ```
   cd cocoapi/PythonAPI&&python3 setup.py install --user
   ```
   
4. Init output(training model output directory) and log(visual log directory) directory:

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
The training info like:
```
2021-12-14 09:45:59     Epoch: [61][0/695]      Time 1.030s (1.030s)    Speed 31.1 samples/s    Data 0.819s (0.819s)    Lr 0.0010 (0.0010)      Loss 0.00045 (0.00045)  Accuracy 0.851 (0.851)
2021-12-14 09:46:49     Epoch: [61][50/695]     Time 1.268s (0.992s)    Speed 25.2 samples/s    Data 1.026s (0.800s)    Lr 0.0010 (0.0010)      Loss 0.00044 (0.00047)  Accuracy 0.878 (0.849)
```

### Valid on MPII using pretrained models

```
python pose_estimation/valid.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file output/mpii/pose_resnet_50/256x256_d256x3_adam_lr1e-3/model_best.pdparams
```
The valid info like:
```
Test: [0/93]    Time 0.774 (0.774)      Loss 0.0004 (0.0004)    Accuracy 0.909 (0.909)
Test: [50/93]   Time 1.108 (0.790)      Loss 0.0005 (0.0005)    Accuracy 0.837 (0.852)
| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
|---|---|---|---|---|---|---|---|---|---|
| 256x256_pose_resnet_50_d256d256d256 | 94.884 | 92.646 | 82.990 | 76.239 | 83.815 | 76.224 | 70.382 | 83.302 | 27.263 |
```
### Export .ONNX using pretrained models

```
python pose_estimation/export.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml \
    --model-file output/mpii/pose_resnet_50/256x256_d256x3_adam_lr1e-3/model_best.pdparams
```
The exporting info like:
```
2021-12-14 09:57:14 [INFO]      ONNX model saved in ./output/onnx/posenet.onnx
=> Model saved as: ./output/onnx/posenet.onnx
=> Done.
```

### Predict with exported ONNX models
```
python pose_estimation/predict.py --img data/mpii/images/004645041.jpg --model output/onnx/posenet.onnx --type ONNX --width 1280 --height 720 
```
### Citation

Original project: [https://github.com/microsoft/human-pose-estimation.pytorch](https://github.com/microsoft/human-pose-estimation.pytorch)

If you want to cite the work in your research, please cite with:
```
@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}
```
