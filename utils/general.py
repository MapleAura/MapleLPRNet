#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import logging
import os
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

from model.lprnet import CHARS

logger = logging.getLogger(__name__)


def set_logging():
    logging.basicConfig(format="%(message)s", level=logging.INFO)


def increment_dir(dir_name, comment=''):
    # Increments a directory runs/exp1 --> runs/exp2_comment
    n = 0  # number
    dir_name = str(Path(dir_name))  # os-agnostic
    d = sorted(glob.glob(dir_name + '*'))  # directories
    if len(d):
        n = max([int(x[len(dir_name):x.find('_') if '_' in x else None]) for x in d]) + 1  # increment
    return dir_name + str(n) + ('_' + comment if comment else '')


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def plot_images(images, fname='images.jpg'):  # TODO labels
    if os.path.isfile(fname):  # do not overwrite
        return None

    images = images.cpu().numpy()

    # un-normalise
    images /= .0078431
    images += 127.5

    bs, _, h, w = images.shape  # batch size, _, height, width
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    for i, img in enumerate(images):
        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=1)

    if fname is not None:
        cv2.imwrite(fname, mosaic)  # , cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

    return mosaic


def model_info(model):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients

    try:  # FLOPS
        from thop import profile
        flops = profile(deepcopy(model), inputs=(torch.zeros(1, 3, 24, 94),), verbose=False)[0] / 1E9 * 2
        fs = ', %.1f GFLOPS' % (flops * 100)
    except:
        fs = ''

    logger.info('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))


def decode(preds):
    last_chars_idx = len(CHARS) - 1

    # greedy decode
    pred_labels = []
    labels = []
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        pred_label = []
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        no_repeat_blank_label = []
        pre_c = -1
        for c in pred_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == last_chars_idx):
                if c == last_chars_idx:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank_label)

    for _, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        labels.append(lb)

    return labels, pred_labels


# class MultiModelWrapper(torch.nn.ModuleList):
#     def __init__(self, models):
#         super(MultiModelWrapper, self).__init__()
#         for i, model in enumerate(models):
#             self.append(model)
#
#     def forward(self, x):
#         for module in self:
#             x = module(x)
#         return x


def sparse_tuple_for_ctc(lpr_max_len, lengths):
    input_lengths = []
    target_lengths = []

    for length in lengths:
        input_lengths.append(lpr_max_len)
        target_lengths.append(length)

    return tuple(input_lengths), tuple(target_lengths)
