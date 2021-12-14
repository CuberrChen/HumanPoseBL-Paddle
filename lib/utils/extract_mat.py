#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/14 下午8:14
# @Author  : chenxb
# @FileName: extract_mat.py
# @Software: PyCharm
# @mail ：joyful_chen@163.com


import scipy.io as scio

gt_valid = scio.loadmat('../../data/mpii/annot/gt_valid.mat')
gt_valid['jnt_missing'] = gt_valid['jnt_missing'][:,:8]
gt_valid['pos_pred_src'] = gt_valid['pos_pred_src'][:,:,:8]
gt_valid['pos_gt_src'] = gt_valid['pos_gt_src'][:,:,:8]
gt_valid['headboxes_src'] = gt_valid['headboxes_src'][:,:,:8]
scio.savemat('../../data/mpii/annot/gt_valid.mat', gt_valid)
