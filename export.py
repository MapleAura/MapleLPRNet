#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import onnx  # noqa: F401  必须在 torch 之前导入: 先加载 protobuf C 扩展可固定使用 anaconda 自带的
# libstdc++(含 GLIBCXX_3.4.29),避免之后 torch 先拉起系统旧版 /lib64/libstdc++.so.6 导致
# "GLIBCXX_3.4.29 not found" 导入失败。
import argparse
import logging

import torch
import torch.nn as nn

from model.lprnet import LPRNetV2, CHARS
from utils.general import resolve_head_cfg, set_logging

logger = logging.getLogger(__name__)
set_logging()


class Uint8Preprocess(nn.Module):
    # 把减均值除以方差的预处理写入模型: 输入 uint8 NHWC 原图,
    # 内部先转 NCHW, 再与训练时一致做 (x - mean) / std 归一化
    def __init__(self, mean=127.5, std=127.5):
        super(Uint8Preprocess, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = x.to(torch.float32)  # uint8 -> float32
        return (x - self.mean) / self.std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/media/code/MapleLPRNet/runs/exp10/weights/best.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--img-size', default=(128, 48), type=lambda s: tuple(map(int, s.split(','))),
                        help='the image size, e.g. 160,48')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--width-mult', type=float, default=1,
                        help='LPRNetV2 channel width multiplier, auto detected from checkpoint when mismatched.')
    parser.add_argument('--grid-h', type=int, default=4,
                        help='head grid height, only used when the checkpoint has no head_cfg (old checkpoints).')
    parser.add_argument('--grid-w', type=int, default=27,
                        help='head grid width / CTC time steps, only used when the checkpoint has no head_cfg (old checkpoints).')
    parser.add_argument('--onnxruntime', action='store_true',
                        help='export for onnxruntime: uint8 input, normalization is embedded in the model.')
    opts = parser.parse_args()

    # 打印参数
    logger.info("args: %s" % opts)

    # Input: onnxruntime 模式下输入 uint8 NHWC 原图,预处理在模型内部完成
    if opts.onnxruntime:
        img = torch.zeros((opts.batch_size, opts.img_size[1], opts.img_size[0], 3), dtype=torch.uint8)
    else:
        img = torch.zeros((opts.batch_size, 3, opts.img_size[1], opts.img_size[0]), dtype=torch.float32)

    # Load weights
    device = torch.device('cpu')
    ckpt = torch.load(opts.weights, map_location=device)

    # 定义网络(自动识别 checkpoint 中的 width_mult)
    width_candidates = []
    if 'width_mult' in ckpt:
        width_candidates.append(ckpt['width_mult'])
    for w in (opts.width_mult, 1.0):
        if w not in width_candidates:
            width_candidates.append(w)
    head_cfg = resolve_head_cfg(ckpt.get('head_cfg'), opts.grid_h, opts.grid_w)
    if head_cfg:
        logger.info('Head config from checkpoint: %s' % head_cfg)
    model = None
    last_err = None
    for w in width_candidates:
        try:
            m = LPRNetV2(8, True, class_num=len(CHARS), dropout_rate=opts.dropout_rate,
                         width_mult=w, img_size=opts.img_size, **head_cfg)
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
    model.eval()
    logger.info('Time steps T=%d (grid_size=%s).' % (int(model.head_cfg['grid_size'][1]), model.head_cfg['grid_size']))
    logger.info("Build network is successful.")

    # onnxruntime 模式: 将减均值除以方差预处理并入模型,导出后输入为 uint8 NHWC
    if opts.onnxruntime:
        model = nn.Sequential(Uint8Preprocess(), model)
        model.eval()
        logger.info('Normalization preprocessing is embedded, input dtype: uint8.')

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
    y = model(img)  # dry run

    # ONNX export
    try:
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opts.weights.replace('.pt', '_ort.onnx' if opts.onnxruntime else '.onnx')  # filename
        torch_major, torch_minor = map(int, torch.__version__.split('.')[:2])
        if (torch_major, torch_minor) >= (2, 4):
            # 固定导出 opset 11:dynamo 导出器需额外安装 onnxscript 且不便限制 opset,
            # 故使用 TorchScript 导出器(dynamo=False);输入尺寸固定时 adaptive_avg_pool2d
            # 会正常分解为 opset 11 兼容算子。
            torch.onnx.export(model, img, f, verbose=False, opset_version=11, dynamo=False,
                              input_names=['images'], output_names=['output'])
        else:
            torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'], output_names=['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('Export complete.')
