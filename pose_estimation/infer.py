from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import cv2
import numpy as np

import paddle
import paddle.optimizer as optim
import paddle.distributed as dist
import paddle.vision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.inference import get_final_preds
from utils.utils import create_logger
from utils.transforms import flip_back, get_affine_transform

import dataset
import models

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    parser.add_argument('--img-file',
                        help='input your test img',
                        type=str,
                        default='')
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--flip-test',
                        help='use flip test',
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    args = parser.parse_args()
    return args


def reset_config(config, args):
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap

def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)


def _xywh2cs(x, y, w, h, image_width, image_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(config)

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_dict(paddle.load(config.TEST.MODEL_FILE))
    else:
        print("The model file must be point out!")
        raise ModuleNotFoundError

    nranks = paddle.distributed.ParallelEnv().nranks
    if nranks>1:
        dist.init_parallel_env()
        model = paddle.DataParallel(model)

    # Confirm flip_pairs Acccording to the dataset class for training model.
    if config.DATASET.DATASET=='mpii':
        flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
    if config.DATASET.DATASET=='coco':
        flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],[9, 10], [11, 12], [13, 14], [15, 16]]
    else:
        flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]

    # Loading an image
    image_file = args.img_file
    data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if data_numpy is None:
        logger.error('=> fail to read {}'.format(image_file))
        raise ValueError('=> fail to read {}'.format(image_file))

    # TODO object detection box; for different image,the box is different.
    # it should be obtained by object detction model or manual box selection.
    box = [0, 0, 280, 380]
    c, s = _box2cs(box, config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1])
    r = 0

    trans = get_affine_transform(c, s, r, config.MODEL.IMAGE_SIZE)
    input = cv2.warpAffine(
        data_numpy,
        trans,
        (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input = transform(input).unsqueeze(0)
    # switch to evaluate mode
    model.eval()
    with paddle.no_grad():
        # compute output heatmap
        output = model(input)
        if config.TEST.FLIP_TEST:
            # this part is ugly, because pypaddle has not supported negative index
            # input_flipped = model(input[:, :, :, ::-1])
            input_flipped = np.flip(input.cpu().numpy(), 3).copy()
            input_flipped = paddle.to_tensor(input_flipped)
            output_flipped = model(input_flipped)
            output_flipped = flip_back(output_flipped.cpu().numpy(), flip_pairs)
            output_flipped = paddle.to_tensor(output_flipped.copy())

            # feature is not aligned, shift flipped heatmap for higher accuracy
            if config.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.clone()[:, :, :, 0:-1]
                # output_flipped[:, :, :, 0] = 0

            output = (output + output_flipped) * 0.5

        preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), np.asarray([c]), np.asarray([s]))

        image = data_numpy.copy()
        for mat in preds[0]:
            x, y = int(mat[0]), int(mat[1])
            cv2.circle(image, (x, y), 2, (255, 0, 0), 2)

            # vis result
        cv2.imwrite("result.jpg", image)
        cv2.imshow('res', image)
        cv2.waitKey(10000)

if __name__ == '__main__':
    main()