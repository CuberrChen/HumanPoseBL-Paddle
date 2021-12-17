# Simple Baselines for Human Pose Estimation and Tracking

## Introduction
This is an paddle implementation of [*Simple Baselines for Human Pose Estimation and Tracking*](https://arxiv.org/abs/1804.06208). 
This work provides baseline methods that are surprisingly simple and effective, thus helpful for inspiring and evaluating new ideas for the field.
State-of-the-art results are achieved on challenging benchmarks.

## Main Results
### Results on MPII val
| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1|
|---|---|---|---|---|---|---|---|---|---|
| 256x256_pose_resnet_50_d256d256d256 | 96.658 | 95.482 | 89.041 | 83.587 | 88.333 | 84.625 | 80.090 | 88.808 | 34.241 |

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
   ├── images
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
The training output info like:
```
2021-12-14 09:45:59     Epoch: [61][0/695]      Time 1.030s (1.030s)    Speed 31.1 samples/s    Data 0.819s (0.819s)    Lr 0.0010 (0.0010)      Loss 0.00045 (0.00045)  Accuracy 0.851 (0.851)
2021-12-14 09:46:49     Epoch: [61][50/695]     Time 1.268s (0.992s)    Speed 25.2 samples/s    Data 1.026s (0.800s)    Lr 0.0010 (0.0010)      Loss 0.00044 (0.00047)  Accuracy 0.878 (0.849)
```

### Valid on MPII using pretrained models

pretained_model link: [https://pan.baidu.com/s/1gaeyGPThltbyeoEQTx7RZQ](https://pan.baidu.com/s/1gaeyGPThltbyeoEQTx7RZQ) valid code: sxfs
```
python pose_estimation/valid.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file output/mpii/pose_resnet_50/256x256_d256x3_adam_lr1e-3/model_best.pdparams
```
The valid info output format as follow:
```
Test: [0/93]	Time 5.032 (5.032)	Loss 0.0003 (0.0003)	Accuracy 0.946 (0.946)
Test: [50/93]	Time 0.254 (0.391)	Loss 0.0004 (0.0004)	Accuracy 0.901 (0.900)
| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
|---|---|---|---|---|---|---|---|---|---|
| 256x256_pose_resnet_50_d256d256d256 | 96.658 | 95.482 | 89.041 | 83.587 | 88.333 | 84.625 | 80.090 | 88.808 | 34.241 |
```
### Infer on Test image using pretrained models
Note: 
**You need to abtain object's location bbox firstly. details in infer.py line 138.**

If there are multiple persons in images, detectors such as Faster R-CNN, SSD or others should be used first to crop them out. Because the simple baseline for human pose estimation is a top-down method.
```
python pose_estimation/infer.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml --img-file images/test.jpg  --model-file output/mpii/pose_resnet_50/256x256_d256x3_adam_lr1e-3/model_best.pdparams
```

### Ultra-fast training-validation using mini datasets

The extracted mpii mini dataset containing 8 training images and 8 validation images is stored in the data folder in the repository.
(The full dataset annotations and images are not included, just to verify the correctness of the training and validation for this project, please replace the entire dataset during subsequent training.)
After installing the dependencies according to the above instructions, ultra-fast training and validation can be performed directly using the following commands:

1. unzip mini datasets
```bash
unzip data/mpii.zip -d data/
```
2. ultra-fast training-validation

```bash
python pose_estimation/train.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml
```
output info on screen as follow:
 ```
=> load 15 samples
=> load 8 samples
2021-12-14 21:15:25     Epoch: [0][0/8] Time 1.835s (1.835s)    Speed 1.1 samples/s     Data 0.073s (0.073s)    Lr 0.0010 (0.0010)      Loss 0.55474 (0.55474)  Accuracy 0.000 (0.000)
Test: [0/1]     Time 1.839 (1.839)      Loss 0.0195 (0.0195)    Accuracy 0.023 (0.023)
| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
|---|---|---|---|---|---|---|---|---|---|
| 256x256_pose_resnet_50_d256d256d256 | 0.000 | 6.250 | 0.000 | 6.250 | 0.000 | 0.000 | 0.000 | 1.852 | 0.000 |
=> saving checkpoint to output/mpii/pose_resnet_50/256x256_d256x3_adam_lr1e-3

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

### Deploy with exported ONNX models
Note: Only support for coco dataset now. The deploy.py is a example only.
```
python pose_estimation/deploy.py --img data/coco/images/xxx.jpg --model output/onnx/posenet.onnx --type ONNX --width 656 --height 384
```
### Citation

Original project: [https://github.com/microsoft/human-pose-estimation.pytorch](https://github.com/microsoft/human-pose-estimation.pytorch)

If you want use other dataset such as coco. Please check the orginal project's experiments setting.

The differences in .yaml file are as follows(**You need to point out the depth of backbone and select lr_sche**):
```
MODEL:
  NAME: pose_resnet
  BACKBONE: resnet
  DEPTH: 50

TRAIN:
  BATCH_SIZE: 128
  SHUFFLE: True
  BEGIN_EPOCH: 0
  END_EPOCH: 140
  RESUME: False
  OPTIMIZER: adam
  LR_TYPE: PolynomialDecay
```

If you want to cite the work in your research, please cite with:
```
@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}
```
