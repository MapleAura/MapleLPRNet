#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import LPRDataSet

from model.lprnet import LPRNetV2, CHARS
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
            flag = False
            if np.array_equal(np.array(pred_labels[j]), label.cpu().numpy()):
                correct_count += 1
                flag = True
            lb = ""
            for i in pred_labels[j]:
                lb += CHARS[i]
            tg = ""
            for k in label:
                tg += CHARS[int(k)]
            print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)

        # Print
        mloss = (mloss * i + loss.item()) / (i + 1)  # update mean losses
        process_count += len(lengths)
        acc = float(correct_count) / float(process_count)
        pbar.set_description('Test mloss: %.5f, macc: %.5f' % (mloss, acc))

    acc = float(correct_count) / float(len(dataset))

    model.float()

    return mloss, acc


def main(opts):
    # 选择设备
    device = torch.device("cuda:0" if (not opts.cpu and torch.cuda.is_available()) else "cpu")
    cuda = device.type != 'cpu'
    logger.info('Use device %s.' % device)

    # 定义网络
    model = LPRNetV2(8, True, class_num=len(CHARS), dropout_rate=opts.dropout_rate).to(device)
    logger.info("Build network is successful.")

    # 损失函数
    ctc_loss = torch.nn.CTCLoss(blank=len(CHARS) - 1, reduction='mean')  # reduction: 'none' | 'mean' | 'sum'

    # Load weights
    ckpt = torch.load(opts.weights, map_location=device)

    # 加载网络
    model.load_state_dict(ckpt["model"])

    # 释放内存
    del ckpt

    # Print
    logger.info('Load weights completed.')

    # 加载数据
    test_dataset = LPRDataSet(args.test_img_dirs.split(","), opts.img_size)
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers,
                             pin_memory=cuda, collate_fn=test_dataset.collate_fn)

    logger.info('Image sizes %d test' % (len(test_dataset)))
    logger.info('Using %d dataloader workers' % opts.workers)

    model.eval()
    
    if args.onnx:
        # 1,3,24,94
        image = torch.ones((1,3,24,94), dtype=torch.float32).cuda()
        prebs = model(image)
        torch.onnx.export(model, image, "lprnet.onnx",  
            input_names=['in'], output_names=['out'], opset_version=12) 
        
    test(model, test_loader, test_dataset, device, ctc_loss, opts.lpr_max_len, opts.float_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STNet & LPRNet Testing')
    parser.add_argument('--test_img_dirs', default="/mnt/code/0.dataset/plate/test.txt", help='the test images path')
    parser.add_argument('--weights',          type=str,            default="/mnt/code/13.lprnet/LPRNet/runs/exp30/weights/best.pt",     help='initial weights path.')
    parser.add_argument('--img-size', default=(94, 24), help='the image size')
    parser.add_argument('--cpu',              action='store_true',                    help='force use cpu.')
    parser.add_argument('--batch-size',       type=int,            default=128,       help='train batch size.')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--lpr-max-len', default=18, help='license plate number max length.')
    parser.add_argument('--float-test',       action='store_true',                    help='use float model run test.')
    parser.add_argument('--workers',          type=int,            default=8,        help='maximum number of dataloader workers.')
    parser.add_argument('--worker-dir',       type=str,            default='runs',    help='worker dir.')
    parser.add_argument('--onnx', default=True, type=bool, help='show test image and its predict result or not.')
    args = parser.parse_args()
    del parser

    # 打印参数
    logger.info("args: %s" % args)

    # 自动调整的参数(不打印)
    args.cache_dir = os.path.join(args.worker_dir, 'cache')

    # 参数处理后的初始化工作
    os.makedirs(args.cache_dir, exist_ok=True)

    # 开始训练
    main(args)
