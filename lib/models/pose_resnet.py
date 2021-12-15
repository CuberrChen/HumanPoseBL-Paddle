#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/12 下午2:38
# @Author  : chenxb
# @FileName: pose_resnet.py
# @Software: PyCharm
# @mail ：joyful_chen@163.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import paddle
import paddle.nn as nn
from collections import OrderedDict

import utils.utils as utils
from models.backbones.resnet import get_resnet

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class PoseResNet(nn.Layer):
    """
    backbone: nn.Layers
    cfg: config
    """

    def __init__(self, backbone, cfg):
        super(PoseResNet, self).__init__()
        self.inplanes = 2048
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.backbone = backbone
        self.pretrained = cfg.MODEL.PRETRAINED if cfg.MODEL.PRETRAINED != '' else None
        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2D(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.Conv2DTranspose(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias_attr=self.deconv_with_bias))
            layers.append(nn.BatchNorm2D(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU())
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def init_weight(self):
        if self.pretrained:
            utils.load_pretrained_model(self, self.pretrained)


def get_pose_net(cfg, is_train):
    backbone = get_resnet(depth=cfg.MODEL.DEPTH)
    model = PoseResNet(backbone=backbone, cfg=cfg)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weight()
    # print(model)
    return model
