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
from utils.general import decode, sparse_tuple_for_ctc, resolve_head_cfg, count_adjacent_repeats, set_logging

logger = logging.getLogger(__name__)
set_logging()


def test(model, data_loader, dataset, device, ctc_loss, input_len, float_test=False, dump_errors=None):
    correct_count = 0
    process_count = 0
    repeat_total = 0
    repeat_correct = 0
    repeat_deleted = 0

    half = not float_test and (device.type != 'cpu')
    if half:
        model.half()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc='Test')
    mloss = 0.0
    sample_idx = 0  # 验证集顺序遍历(shuffle=False),用它把错例映射回图片路径
    err_file = open(dump_errors, 'w') if dump_errors else None
    for i, (imgs, labels, lengths) in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        imgs = imgs.half() if half else imgs.float()
        labels = labels.half() if half else labels.float()

        # 准备 loss 计算的参数
        input_lengths, target_lengths = sparse_tuple_for_ctc(input_len, lengths)

        with torch.no_grad():
            x = model(imgs)
            y = x.permute(2, 0, 1)  # [batch_size, chars, width] -> [width, batch_size, chars]
            y = y.log_softmax(2)
            loss = ctc_loss(y.float(), labels.float(), input_lengths=input_lengths, target_lengths=target_lengths)

        x = x.cpu().detach().numpy()
        _, pred_labels = decode(x)

        start = 0
        for j, length in enumerate(lengths):
            label = labels[start:start + length]
            start += length
            gt = label.cpu().numpy()
            pred = np.array(pred_labels[j])
            flag = False
            if np.array_equal(pred, gt):
                correct_count += 1
                flag = True
            # 相邻重复字符子集: greedy CTC 必须在两个相同字符之间吐 blank 才能解出
            # "663" 这类连号, 单独统计该子集准确率与"被吞字符"(预测比 GT 短)的次数。
            if count_adjacent_repeats(gt):
                repeat_total += 1
                if flag:
                    repeat_correct += 1
                if len(pred) < len(gt):
                    repeat_deleted += 1
            # Removed per-sample print output
            lb = ""
            for ci in pred_labels[j]:
                lb += CHARS[ci]
            tg = ""
            for k in label:
                tg += CHARS[int(k)]
            # print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)
            if err_file is not None and not flag:
                # 难例清单(给 train.py --hard-list 用): "<图片路径> <真值> <预测>"
                err_file.write('%s %s %s\n' % (dataset.img_paths[sample_idx].split('#')[0], tg, lb))
            sample_idx += 1

        # Print
        mloss = (mloss * i + loss.item()) / (i + 1)  # update mean losses
        process_count += len(lengths)
        acc = float(correct_count) / float(process_count)
        pbar.set_description('Test mloss: %.5f, macc: %.5f' % (mloss, acc))

    acc = float(correct_count) / float(len(dataset))
    if err_file is not None:
        err_file.close()
        logger.info('Error cases dumped to %s (%d samples)' % (dump_errors, len(dataset) - correct_count))
    stats = {
        'correct': correct_count,
        'total': len(dataset),
        'repeat_total': repeat_total,
        'repeat_correct': repeat_correct,
        'repeat_deleted': repeat_deleted,
    }

    model.float()

    return mloss, acc, stats


def main(opts):
    # 选择设备
    device = torch.device("cuda:0" if (not opts.cpu and torch.cuda.is_available()) else "cpu")
    cuda = device.type != 'cpu'
    logger.info('Use device %s.' % device)

    # Load weights
    ckpt = torch.load(opts.weights, map_location=device)

    # 定义网络(自动识别 checkpoint 中的 width_mult)
    width_candidates = []
    if 'width_mult' in ckpt:
        width_candidates.append(ckpt['width_mult'])
    for w in (opts.width_mult, 1.0):
        if w not in width_candidates:
            width_candidates.append(w)
    head_cfg = resolve_head_cfg(ckpt.get('head_cfg'), opts.grid_h, opts.grid_w)
    model = None
    last_err = None
    for w in width_candidates:
        try:
            m = LPRNetV2(8, True, class_num=len(CHARS), dropout_rate=opts.dropout_rate,
                         width_mult=w, img_size=opts.img_size, **head_cfg).to(device)
            m.load_state_dict(ckpt["model"])
            model = m
            if w != opts.width_mult:
                logger.info('Auto detect width_mult=%.2f from checkpoint.' % w)
            break
        except RuntimeError as e:
            last_err = e
            continue
    if model is None:
        raise RuntimeError('Failed to load checkpoint %s with width_mult in %s; last error: %s'
                           % (opts.weights, width_candidates, last_err))
    del ckpt
    input_len = int(model.head_cfg['grid_size'][1])
    if opts.lpr_max_len not in (None, input_len):
        logger.warning('--lpr-max-len=%s ignored: CTC input_lengths is derived from grid_size and equals T=%d.'
                       % (opts.lpr_max_len, input_len))
    logger.info('Time steps T=%d (grid_size=%s).' % (input_len, model.head_cfg['grid_size']))
    logger.info("Build network is successful.")

    # 损失函数
    ctc_loss = torch.nn.CTCLoss(blank=len(CHARS) - 1, reduction='mean')  # reduction: 'none' | 'mean' | 'sum'

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
        # 1,3,H,W (与 opts.img_size 对应,默认 146x48)
        image = torch.ones((1, 3, opts.img_size[1], opts.img_size[0]), dtype=torch.float32).to(device)
        prebs = model(image)
        torch.onnx.export(model, image, "lprnet.onnx",
            input_names=['in'], output_names=['out'], opset_version=12)
        
    mloss, acc, stats = test(model, test_loader, test_dataset, device, ctc_loss, input_len,
                             opts.float_test, opts.dump_errors or None)
    rep_n = stats['repeat_total']
    rep_acc = float(stats['repeat_correct']) / rep_n if rep_n else 0.0
    logger.info('Test mloss: %.5f, macc: %.5f (%d/%d)' % (mloss, acc, stats['correct'], stats['total']))
    logger.info('Repeat subset: acc=%.5f (n=%d, correct=%d, deleted=%d)'
                % (rep_acc, rep_n, stats['repeat_correct'], stats['repeat_deleted']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STNet & LPRNet Testing')
    parser.add_argument('--test_img_dirs', default="/mnt/workspace/xxx/data/ccpd_val.txt", help='the test images path')
    parser.add_argument('--weights',          type=str,            default="/mnt/workspace/xxx/code/LPRNet/runs/exp17/weights/best.pt",     help='initial weights path.')
    parser.add_argument('--img-size', default=(128, 48), type=lambda s: tuple(map(int, s.split(','))),
                        help='the image size, e.g. 128,48')
    parser.add_argument('--cpu',              action='store_true',                    help='force use cpu.')
    parser.add_argument('--batch-size',       type=int,            default=128,       help='train batch size.')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--width-mult', type=float, default=1, help='LPRNetV2 channel width multiplier, auto detected from checkpoint when mismatched.')
    parser.add_argument('--grid-h', type=int, default=4, help='head grid height, only used when the checkpoint has no head_cfg (old checkpoints).')
    parser.add_argument('--grid-w', type=int, default=27, help='head grid width / CTC time steps, only used when the checkpoint has no head_cfg (old checkpoints).')
    parser.add_argument('--lpr-max-len', type=int, default=None, help='consistency check only: must equal the checkpoint grid width if set (CTC input_lengths is derived from it).')
    parser.add_argument('--float-test',       action='store_true',                    help='use float model run test.')
    parser.add_argument('--dump-errors', type=str, default='',
                        help='把错例写成 "<图片路径> <真值> <预测>" 到该文件, 可直接喂给 train.py --hard-list.')
    parser.add_argument('--workers',          type=int,            default=8,        help='maximum number of dataloader workers.')
    parser.add_argument('--worker-dir',       type=str,            default='runs',    help='worker dir.')
    parser.add_argument('--onnx', default=False, type=bool, help='show test image and its predict result or not.')
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
