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


def resolve_head_cfg(head_cfg, grid_h=4, grid_w=18):
    """合并命令行网格参数与 checkpoint 中的 head 配置。

    checkpoint 的 head_cfg 带 grid_size 时以 checkpoint 为准(权重就是按该网格训练的),
    仅在缺失时(旧版 checkpoint)才用命令行的 --grid-h/--grid-w 兜底。
    """
    cfg = dict(head_cfg or {})
    if 'grid_size' not in cfg:
        cfg['grid_size'] = (grid_h, grid_w)
    return cfg


def count_adjacent_repeats(labels):
    """相邻相同标签的个数 R。

    greedy CTC 解出连续相同字符必须在两字符之间插入 blank, 因此长 L、相邻重复 R 处的
    车牌至少需要 L + R 帧才能无歧义解码(如 皖A378663: L=8, R=1 -> 至少 9 帧)。
    返回 0 表示该车牌不含相邻重复字符。
    """
    labels = list(labels)
    return sum(1 for a, b in zip(labels, labels[1:]) if a == b)


def shrink_state_dict(model, state_dict, old_width=1.0):
    """Net2Net 式"缩权": 从更宽 checkpoint 里选出子网络, 初始化到更窄模型。

    与 expand_state_dict(扩权)不同, 缩权不是精确等价的: 被裁掉的通道信息直接丢失,
    得到的只是一个不错的初始化, 仍然必须重新训练。

    选择准则: Conv 的输出通道取重要性 top-k(保留索引升序, 便于和旧模型对照)。重要性优先
    用紧随其后的 BatchNorm 的 gamma(|weight|) —— BN 会把卷积核的绝对尺度归一化掉, 此时
    "按卷积核 L2 能量"几乎是噪声; 后面没有 BN 的卷积才退回按卷积核 L2 能量。下一层的输入
    通道严格复用上一层选中的索引, 保证前后通道一一对应; BatchNorm 的 weight/bias/running
    统计按同一索引裁剪; head 分支的输出拼进 container 卷积时按各段偏移拼接保留索引。
    head 结构改版导致无法对齐时跳过该层(保持随机初始化)并在返回值 missing 里报告。

    返回 (copied, shrunk, missing): 精确拷贝的层数、按通道裁剪的层数、未使用的键。
    """
    import torch.nn as nn

    model_state = model.state_dict()
    result = {}
    stats = {'copied': 0, 'shrunk': 0}
    skipped = []

    # 找出每个 Conv 之后紧邻的 BatchNorm(中间只隔 ReLU/Pool/Dropout 等无通道算子),
    # 用它的 gamma(|weight|) 作为该 Conv 输出通道的重要性。
    # 关键: BN 会归一化掉卷积核的绝对尺度, 所以"按卷积核能量"在有 BN 的网络里几乎是
    # 噪声; Network Slimming 用 BN 的 gamma 才是有效的通道重要性判据。
    conv_to_bn = {}
    pending = []  # [(conv_prefix, out_channels)] 尚未被 BN 认领的卷积

    def scan(module, prefix):
        for cname, child in module.named_children():
            cprefix = (prefix + '.' + cname) if prefix else cname
            if isinstance(child, nn.Conv2d):
                pending.append((cprefix, child.out_channels))
            elif isinstance(child, nn.BatchNorm2d):
                if pending and pending[-1][1] == child.num_features:
                    conv_to_bn[pending.pop()[0]] = cprefix
            scan(child, cprefix)

    scan(model, '')

    def importance(conv_prefix, ov):
        bn = conv_to_bn.get(conv_prefix)
        if bn is not None:
            g = state_dict.get(bn + '.weight')
            if g is not None and g.shape[0] == ov.shape[0]:
                return g.detach().float().abs()
        # 后面没有 BN: 退回按卷积核 L2 能量
        return ov.detach().float().reshape(ov.shape[0], -1).pow(2).sum(dim=1)

    def keep_top(imp, k):
        o = imp.shape[0]
        if k >= o:
            return list(range(o))
        return sorted(torch.topk(imp, k).indices.tolist())

    def do_conv(prefix, in_keep):
        name = prefix + '.weight'
        ov, v = state_dict.get(name), model_state.get(name)
        if v is None:
            return in_keep
        if ov is None:
            skipped.append(name)
            return list(range(v.shape[0]))
        bname = prefix + '.bias'
        ob, vb = state_dict.get(bname), model_state.get(bname)
        has_bias = ob is not None and vb is not None
        if ov.shape == v.shape:
            result[name] = ov.clone()
            if has_bias and ob.shape == vb.shape:
                result[bname] = ob.clone()
            stats['copied'] += 1
            return list(range(ov.shape[0]))
        # in_keep 是"旧通道索引"列表, 其长度必须等于新模型的输入通道数
        if v.shape[1] != len(in_keep):
            raise ValueError('shrink mismatch for %s: new in=%d but kept=%d old channels'
                             % (name, v.shape[1], len(in_keep)))
        out_keep = keep_top(importance(prefix, ov), v.shape[0])
        result[name] = ov[out_keep][:, in_keep].clone()
        stats['shrunk'] += 1
        if has_bias:
            result[bname] = ob[out_keep].clone()
        return out_keep

    def do_bn(prefix, in_keep):
        for suffix in ('.weight', '.bias', '.running_mean', '.running_var'):
            name = prefix + suffix
            ov, v = state_dict.get(name), model_state.get(name)
            if v is None:
                continue
            if ov is None:
                skipped.append(name)
                continue
            if ov.shape == v.shape:
                result[name] = ov.clone()
                stats['copied'] += 1
            else:
                result[name] = ov[in_keep].clone()
                stats['shrunk'] += 1
        name = prefix + '.num_batches_tracked'
        if state_dict.get(name) is not None and model_state.get(name) is not None:
            result[name] = state_dict[name].clone()

    def walk(module, prefix, in_keep):
        if isinstance(module, nn.Conv2d):
            return do_conv(prefix, in_keep)
        if isinstance(module, nn.BatchNorm2d):
            do_bn(prefix, in_keep)
            return in_keep
        keep = in_keep
        for cname, child in module.named_children():
            keep = walk(child, (prefix + '.' + cname) if prefix else cname, keep)
        return keep

    # backbone: 顺序链, 逐 child 传递输出的保留通道索引
    keep = [0, 1, 2]  # 输入 3 通道全保留
    keep_at = {}
    for i, child in enumerate(model.backbone.children()):
        keep = walk(child, 'backbone.%d' % i, keep)
        keep_at[i] = keep

    # head 分支: 输入是 backbone 对应 keep_idx 处的特征
    new_groups = list(getattr(model, 'head_group_chs', ()))
    keep_idx = list(getattr(model, 'keep_idx', ()))
    head_out_keep = []
    for bi, head in enumerate(model.heads):
        fallback = list(range(new_groups[bi])) if bi < len(new_groups) else []
        keep = keep_at.get(keep_idx[bi], fallback) if bi < len(keep_idx) else fallback
        for cname, child in head.named_children():
            keep = walk(child, 'heads.%d.%s' % (bi, cname), keep)
        head_out_keep.append(keep)

    # container: 输入是各 head 分支输出 concat, 需要按旧模型的分段偏移拼接保留索引。
    # 旧 head 的输出通道优先从 checkpoint 里读(head_ch 是配置项, 不随 width_mult 变),
    # 只有"纯池化无参数"的 head 才退回到宽度推导的 backbone 特征通道数。
    width_groups = [int(round(64 * old_width)), int(round(128 * old_width)),
                    int(round(256 * old_width)), int(model.class_num)]
    old_groups = []
    for bi in range(len(head_out_keep)):
        g = None
        for name in ('heads.%d.0.weight' % bi, 'heads.%d.1.weight' % bi):
            t = state_dict.get(name)
            if t is not None:
                g = int(t.shape[0])
                break
        if g is None and bi < len(width_groups):
            g = width_groups[bi]
        old_groups.append(g)
    ovc, vc = state_dict.get('container.0.weight'), model_state.get('container.0.weight')
    aligned = (ovc is not None and vc is not None and ovc.shape[1] == sum(old_groups)
               and len(head_out_keep) == len(old_groups) and new_groups == [len(k) for k in head_out_keep]
               and all((not k) or max(k) < g for k, g in zip(head_out_keep, old_groups)))
    if aligned:
        concat_keep = []
        off = 0
        for g, k in zip(old_groups, head_out_keep):
            concat_keep += [off + i for i in k]
            off += g
        do_conv('container.0', concat_keep)  # 输出通道不变, 只按输入裁剪
    else:
        skipped.append('container.0.weight')

    model.load_state_dict(result, strict=False)
    missing = sorted((set(model_state) - set(result)) | set(skipped))
    return stats['copied'], stats['shrunk'], missing


def recalibrate_bn(model, data_loader, device, max_batches=20):
    """缩权后重估 BatchNorm 的 running 统计。

    结构化剪枝会改变各层激活的分布(尤其是"输入通道被裁掉"的卷积, 其输出已经不等于原来
    那条通道的输出), 直接沿用 checkpoint 里的 running_mean/var 会让 BN 输出被放大若干倍。
    这里只用训练数据做前向, 把每层 running 统计重算成校准集上的真实均值/方差:
    reset_running_stats() + momentum=None(累计平均), 只把 BN 置为 train, 其余保持 eval
    (避免 Dropout 干扰统计)。

    返回实际用于校准的 batch 数。
    """
    import torch.nn as nn

    was_training = model.training
    model.eval()
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    momenta = {}
    for m in bns:
        m.reset_running_stats()
        momenta[m] = m.momentum
        m.momentum = None  # 累计平均 -> 等价于校准集上的精确均值/方差
        m.train()  # eval 模式下 BN 不会更新 running 统计

    n = 0
    with torch.no_grad():
        for batch in data_loader:
            imgs = batch[0].to(device, non_blocking=True).float()
            model(imgs)
            n += 1
            if n >= max_batches:
                break

    for m in bns:
        m.momentum = momenta[m]
    model.train(was_training)
    return n


def expand_state_dict(model, state_dict, old_width=1.0, jitter=0.001):
    """Net2Net 式扩权: 把窄模型权重复制进更宽模型, 扩权后前向与旧模型基本一致.

    结构说明(LPRNetV2): head 只在 backbone 特征上做池化 + BN, 没有跨层连接,
    因此新增通道不参与旧通道的计算, 置零即可(也不会被旧层放大).

    策略:
    - Conv: 旧输出通道 1:1 拷贝; 新增输出通道按环形复制旧通道权重并加微小抖动,
      新增输入通道(列)置 0(该列特征不影响任何旧通道, 但梯度可流入);
    - BN: weight/bias/running 统计按同样的环形复制(新增通道行为与旧通道一致);
    - container 卷积输入是各 head 分支 concat: 新旧 head 通道布局一致时按组偏移
      放置旧权重(组内新增列置 0); head 结构改版时旧 readout 对新特征无意义,
      该层跳过(保持随机初始化), 并在 missing 里报告;
    - 这样前向输出 ≈ 旧模型, 训练时新通道(复制+抖动)与旧通道近似, 可稳定继续训练.
    - 新模型有、checkpoint 没有的层(head 改版后新增的 BN 等)保持随机初始化,
      只用 checkpoint 做部分初始化时也能正常训练.
    """
    import torch.nn as nn
    model_state = model.state_dict()
    mods = dict(model.named_modules())
    copied = 0
    expanded = 0
    group_old = group_new = None
    if hasattr(model, 'class_num') and hasattr(model, 'width_mult'):
        group_old = [int(round(64 * old_width)), int(round(128 * old_width)),
                     int(round(256 * old_width)), int(model.class_num)]
        group_new = [int(round(64 * model.width_mult)), int(round(128 * model.width_mult)),
                     int(round(256 * model.width_mult)), int(model.class_num)]
    skip = []
    for k, v in model_state.items():
        if k not in state_dict:
            continue
        ov = state_dict[k]
        if ov.shape == v.shape:
            model_state[k] = ov.clone()
            copied += 1
            continue
        if k == 'container.0.weight' and group_old is not None and group_new is not None:
            # container 的输入是各 head 分支 concat: 只有新旧 head 通道布局一致时才能按组
            # 搬权重。head 改版(如去掉大核 depthwise 卷积、改用池化 + 1x1/3x3 卷积)后
            # 旧 readout 对新特征没有意义, 直接跳过(保持随机初始化, 交给训练学)。
            head_groups = list(getattr(model, 'head_group_chs', None) or group_new)
            if head_groups != list(group_new) or sum(group_old) != ov.shape[1] \
                    or sum(group_new) != v.shape[1]:
                skip.append(k)
                continue
        if len(ov.shape) != len(v.shape):
            raise ValueError('rank mismatch for %s: %s vs %s' % (k, tuple(ov.shape), tuple(v.shape)))
        if any(a > b for a, b in zip(ov.shape, v.shape)):
            raise ValueError('cannot expand %s: %s -> %s (target model is not wider)' % (k, tuple(ov.shape), tuple(v.shape)))
        parent, attr = k.rsplit('.', 1)
        mod = mods.get(parent)
        nv = v.clone()
        if isinstance(mod, nn.Conv2d) and len(ov.shape) == 4:
            out_o, in_o = ov.shape[0], ov.shape[1]
            out_n, in_n = nv.shape[0], nv.shape[1]
            nv.zero_()
            if k == 'container.0.weight' and group_old is not None and sum(group_old) == ov.shape[1]:
                off_o = 0
                off_n = 0
                for go, gn in zip(group_old, group_new):
                    nv[:, off_n:off_n + go] = ov[:, off_o:off_o + go]
                    off_o += go
                    off_n += gn
            else:
                # 旧输出通道/旧输入通道 1:1
                nv[:out_o, :in_o] = ov
                # 新增输出通道: 环形复制旧通道 + 微小抖动(能量与旧通道一致, 打破对称)
                if out_n > out_o:
                    for i in range(out_o, out_n):
                        src = (i - out_o) % out_o
                        nv[i, :in_o] = ov[src]
                        nv[i, :in_o] = nv[i, :in_o] * (1 + jitter * torch.randn_like(nv[i, :in_o]))
                # 新增输入通道保持 0: 不干扰旧通道, 仅提供梯度通路
        elif len(ov.shape) == 1 and isinstance(mod, nn.Conv2d):
            # conv bias: 与输出通道同样环形复制
            o = ov.shape[0]
            nv[:o] = ov
            for i in range(o, nv.shape[0]):
                src = (i - o) % o
                nv[i] = ov[src] * (1 + jitter * torch.randn_like(ov[src]))
        elif len(ov.shape) == 1 and isinstance(mod, nn.BatchNorm2d):
            o = ov.shape[0]
            nv[:o] = ov
            for i in range(o, nv.shape[0]):
                src = (i - o) % o
                nv[i] = ov[src]
                if attr in ('weight', 'bias'):
                    nv[i] = nv[i] * (1 + jitter * torch.randn_like(nv[i]))
        else:
            raise ValueError('unsupported expanded tensor %s (%s -> %s)' % (k, tuple(ov.shape), tuple(v.shape)))
        model_state[k] = nv
        expanded += 1
    missing = sorted((set(state_dict) - set(model_state)) | set(skip))
    model.load_state_dict(model_state, strict=False)
    return copied, expanded, missing
