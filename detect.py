#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import cv2

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import LPRDataSet
from model.lprnet import LPRNet, CHARS
from utils.general import decode, sparse_tuple_for_ctc, set_logging

logger = logging.getLogger(__name__)
set_logging()


def test(model, data_loader, dataset, device, ctc_loss, lpr_max_len, float_test=False):
    correct_count = 0
    process_count = 0

    half = not float_test and (device.type != 'cpu')
    if half:
        model.half()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc='Test')
    mloss = 0.0
    for i, (imgs, labels, lengths) in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        imgs = imgs.half() if half else imgs.float()
        labels = labels.half() if half else labels.float()

        # 准备 loss 计算的参数
        input_lengths, target_lengths = sparse_tuple_for_ctc(lpr_max_len, lengths)

        with torch.no_grad():
            x = model(imgs)
            y = x.permute(2, 0, 1)  # [batch_size, chars, width] -> [width, batch_size, chars]
            y = y.log_softmax(2).requires_grad_()
            loss = ctc_loss(y.float(), labels.float(), input_lengths=input_lengths, target_lengths=target_lengths)

        x = x.cpu().detach().numpy()
        _, pred_labels = decode(x)

        start = 0
        for j, length in enumerate(lengths):
            label = labels[start:start + length]
            start += length
            if np.array_equal(np.array(pred_labels[j]), label.cpu().numpy()):
                correct_count += 1

        # Print
        mloss = (mloss * i + loss.item()) / (i + 1)  # update mean losses
        process_count += len(lengths)
        acc = float(correct_count) / float(process_count)
        pbar.set_description('Test mloss: %.5f, macc: %.5f' % (mloss, acc))

    acc = float(correct_count) / float(len(dataset))

    model.float()

    return mloss, acc


def load_image(file, img_size):
    image = cv2.imread(file)

    # 缩放
    image = cv2.resize(image, img_size)[:, :, ::-1]

    # 归一化
    image = (image.astype('float32') - 127.5) * 0.007843

    # to tensor
    image = torch.from_numpy(image.transpose((2, 0, 1))).contiguous()

    return image


def main(opts):
    # 选择设备
    device = torch.device("cuda:0" if (not opts.cpu and torch.cuda.is_available()) else "cpu")
    logger.info('Use device %s.' % device)

    # 定义网络
    model = LPRNet(class_num=len(CHARS), dropout_rate=opts.dropout_rate).to(device)
    logger.info("Build network is successful.")

    # Load weights
    ckpt = torch.load(opts.weights, map_location=device)

    # 加载网络
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Print
    logger.info('Load weights completed.')

    for name in os.listdir(opts.source_dir):
        filepath = os.path.join(opts.source_dir, name)
        image = load_image(filepath, opts.img_size).unsqueeze(0).to(device)

        with torch.no_grad():
            x = model(image).cpu().detach().numpy()
            pred_labels, _ = decode(x)

        logger.info('{} ------> {}'.format(name, pred_labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPRNet detecting')
    parser.add_argument('--source-dir', type=str, default="figures", help='train images source dir.')
    parser.add_argument('--weights', type=str, default="", help='initial weights path.')
    parser.add_argument('--img-size', default=(94, 24), help='the image size')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--cpu', action='store_true', help='force use cpu.')
    parser.add_argument('--lpr-max-len', default=18, help='license plate number max length.')
    args = parser.parse_args()

    # 打印参数
    logger.info("args: %s" % args)

    # 开始训练
    main(args)
