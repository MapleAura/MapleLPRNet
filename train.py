#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import logging
import math
import os
import time
from pathlib import Path
from data.dataset import LPRDataSet
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.lprnet import LPRNetV2, CHARS
from test import test
from utils.general import increment_dir, plot_images, model_info, set_logging, \
    sparse_tuple_for_ctc, select_device, expand_state_dict, shrink_state_dict, recalibrate_bn

logger = logging.getLogger(__name__)
set_logging()

# 每个 epoch 往 results.csv 追加一行的列定义(改这里时注意兼容旧文件)
RESULTS_HEADER = ['epoch', 'lr', 'train_loss', 'train_loss_weighted', 'val_loss', 'macc',
                  'repeat_n', 'repeat_correct', 'repeat_deleted', 'best_acc', 'is_best',
                  'epoch_time_s', 'total_time_s']


def save_run_config(out_dir, opts, extra=None):
    """把本次训练用的配置落盘成 args.json, 便于事后复现。

    args 里有 tuple(img_size 等), 统一用 default=str 转成字符串。
    """
    config = {'date': time.strftime('%Y-%m-%d %H:%M:%S'),
              'host': os.environ.get('HOSTNAME', ''),
              'out_dir': os.path.abspath(out_dir)}
    config.update(vars(opts))
    if extra:
        config.update(extra)
    path = os.path.join(out_dir, 'args.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)
    return path


def append_results_csv(path, row):
    """每个 epoch 追加一行结果; 文件不存在/为空时先写表头, 写完立即 flush 防中断丢数据。"""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, extrasaction='ignore')
        if new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def adjacent_repeat_counts(labels, target_lengths, device):
    """每个样本的相邻重复字符个数 R(连号加权用, 全程在 device 上算)。

    greedy CTC 要把 "11" 解成两个字符, 必须在两帧之间插入一帧 blank, 所以相邻重复越多
    的样本越难, 训练时给它更大权重。labels 是 collate 拼接后的 1D 张量, target_lengths
    是各样本的目标长度。
    """
    n = labels.numel()
    b = len(target_lengths)
    counts = torch.zeros(b, device=device)
    if n < 2 or b == 0:
        return counts
    ends = torch.cumsum(torch.as_tensor(list(target_lengths), dtype=torch.long, device=device), 0)
    same = labels[1:] == labels[:-1]
    same[ends[:-1] - 1] = False  # 每个样本的最后一位与下一个样本的首位不算相邻
    pair_sample = torch.searchsorted(ends, torch.arange(n - 1, device=device), right=True)
    counts.index_add_(0, pair_sample, same.to(counts.dtype))
    return counts


def compute_loss(ctc_loss, ctc_loss_none, x, labels, input_lengths, target_lengths,
                 sample_w, repeat_weight, device):
    """返回 (反传用的 loss, 打印用的未加权 loss)。

    - 没开任何加权时走原来的 reduction='mean' 路径, 与改动前完全一致;
    - 加权时用 reduction='none' 的逐样本 loss, 先按目标长度归一(与 mean 口径一致), 再按
      样本权重求加权平均。权重按均值归一, 只改变样本间的相对权重, 学习率不用改。
    """
    if repeat_weight <= 0 and sample_w is None:
        loss = ctc_loss(x, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        return loss, loss

    with torch.cuda.amp.autocast(enabled=False):
        # 逐样本 loss 用 fp32 算, 避免半精度下加权/归一化引入额外误差
        per_sample = ctc_loss_none(x.float(), labels, input_lengths=input_lengths,
                                   target_lengths=target_lengths)
        tgt_len = torch.as_tensor(list(target_lengths), device=device, dtype=per_sample.dtype)
        per_sample = per_sample / tgt_len
        w = torch.ones_like(per_sample) if sample_w is None else \
            sample_w.to(device=per_sample.device, dtype=per_sample.dtype)
        if repeat_weight > 0:
            w = w * (1.0 + repeat_weight * adjacent_repeat_counts(labels, target_lengths, device))
        w = w / w.mean()
        loss = (per_sample * w).mean()
        return loss, per_sample.mean()


def log_repeat_stats(dataset, repeat_weight):
    """统计训练集里连号样本占比与加权后的权重分布(调 --repeat-weight 时看这一行)。"""
    repeats = []
    for path in dataset.img_paths:
        label = path.split('#')[1]
        repeats.append(sum(1 for a, b in zip(label, label[1:]) if a == b))
    n = len(repeats)
    hit = sum(1 for r in repeats if r > 0)
    weights = [1.0 + repeat_weight * r for r in repeats]
    logger.info('Repeat stats: %d/%d (%.1f%%) samples contain adjacent repeats; '
                'weight=1+%.2f*R -> min=%.2f, max=%.2f, mean=%.2f (loss 里按均值归一到 1)'
                % (hit, n, 100.0 * hit / max(n, 1), repeat_weight,
                   min(weights), max(weights), sum(weights) / max(n, 1)))


def main(opts):
    epochs = opts.epochs

    # 选择设备
    device = select_device(opts.device, opts.batch_size)
    cuda = device.type != 'cpu'
    logger.info('Use device %s.' % device)

    # 语义: --weights=预训练初始化(从 epoch1 从头训练); --resume=接着训练(恢复 epoch/优化器)
    if opts.weights and opts.resume:
        raise ValueError('Specify either --weights (init, from epoch 1) or --resume (continue training), not both.')
    ckpt_path = opts.resume if opts.resume else opts.weights
    is_resume = bool(opts.resume)

    # 预检: 用 --weights 做初始化且宽度发生变化(扩权/缩权)时, 用较小默认 lr 防止发散
    expand_init = False
    shrink_init = False
    if ckpt_path and not is_resume:
        _ck = torch.load(ckpt_path, map_location='cpu')
        _ck_width = _ck.get('width_mult', 1.0)
        expand_init = opts.width_mult > _ck_width + 1e-6
        shrink_init = opts.width_mult < _ck_width - 1e-6
        del _ck
    if opts.lr is None:
        opts.lr = 0.0003 if (expand_init or shrink_init) else 0.001
        _why = 'expand init' if expand_init else ('shrink init' if shrink_init else 'standard')
        logger.info('Default lr=%.5f (%s).' % (opts.lr, _why))

    # 定义网络
    model = LPRNetV2(8, True, class_num=len(CHARS), dropout_rate=opts.dropout_rate,
                     width_mult=opts.width_mult, img_size=opts.img_size,
                     grid_size=(opts.grid_h, opts.grid_w),
                     head_ch=opts.head_ch, head_ksize=opts.head_ksize,
                     head_pool=opts.head_pool, head_norm=opts.head_norm).to(device)
    logger.info('Head config: %s' % model.head_cfg)
    # CTC 的 input_lengths 必须等于模型实际输出的时间步数 grid_w, 否则短了会静默截断帧、
    # 长了 CTCLoss 直接报错。--grid-w 是唯一入口, --lpr-max-len 只作一致性校验。
    input_len = int(model.head_cfg['grid_size'][1])
    if opts.lpr_max_len not in (None, input_len):
        logger.warning('--lpr-max-len=%s ignored: CTC input_lengths must equal the %d time steps '
                       'of grid_size=%s (use --grid-w to change it).'
                       % (opts.lpr_max_len, input_len, model.head_cfg['grid_size']))
    logger.info('Time steps T=%d (grid_size=%s); a repeat-containing plate of length L needs L+repeats<=T frames.'
                % (input_len, model.head_cfg['grid_size']))
    model_info(model)
    logger.info("Build network is successful.")

    # 优化器
    optimizer_params = [
        {'params': model.parameters(), 'weight_decay': opts.weight_decay}
    ]
    if opts.adam:
        optimizer = torch.optim.Adam(optimizer_params, lr=opts.lr, betas=(opts.momentum, 0.999))
    else:
        optimizer = torch.optim.SGD(optimizer_params, lr=opts.lr, momentum=opts.momentum, nesterov=True)
    del optimizer_params

    # 损失函数
    ctc_loss = torch.nn.CTCLoss(blank=len(CHARS) - 1, reduction='mean')  # reduction: 'none' | 'mean' | 'sum'
    # 连号/难例加权时用的逐样本 loss(reduction='none'); 不加权时不使用, 不影响原行为
    ctc_loss_none = torch.nn.CTCLoss(blank=len(CHARS) - 1, reduction='none')
    if opts.repeat_weight > 0 or opts.hard_list:
        logger.info('Weighted loss enabled: repeat_weight=%.2f, hard_list=%s (hard_weight=%.2f)'
                    % (opts.repeat_weight, opts.hard_list or '-', opts.hard_weight))

    # lr 自动调整器
    lf = lambda e: (((1 + math.cos(e * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    del lf

    # TB
    logger.info('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % opts.worker_dir)
    tb_writer = SummaryWriter(log_dir=opts.out_dir)  # runs/exp0

    # 加载权重: --weights=从头训练初始化; --resume=接着训练
    start_epoch = 1
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_width = ckpt.get('width_mult', 1.0)

        if is_resume:
            # 接着训练: 宽度必须与 checkpoint 一致(恢复 epoch/优化器)
            if abs(ckpt_width - opts.width_mult) > 1e-6:
                raise ValueError('--resume requires the same width_mult: checkpoint=%.2f, --width-mult=%.2f. '
                                 'For width change use --weights <ckpt> which starts from epoch 1.'
                                 % (ckpt_width, opts.width_mult))
            model.load_state_dict(ckpt["model"])
            if ckpt.get('optimizer_type') is not None:
                optimizer_type = 'adam' if opts.adam else 'sgd'
                if optimizer_type == ckpt['optimizer_type']:
                    optimizer.load_state_dict(ckpt['optimizer'])
                else:
                    logger.warning('Optimizer is changed, state has been lost.')
            else:
                logger.warning('Optimizer state missing in checkpoint (e.g. final.pt), state has been lost.')
            start_epoch = ckpt['epoch'] + 1
            if epochs < start_epoch:
                logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                            (ckpt_path, start_epoch - 1, start_epoch + epochs))
                epochs += start_epoch
            logger.info('Resumed training from epoch %d.' % start_epoch)
        else:
            # 从头训练: 仅用 checkpoint 初始化权重, epoch/优化器重新开始
            if opts.width_mult > ckpt_width + 1e-6:
                logger.info('Auto Net2Net-expand init width_mult=%.2f -> %.2f.' % (ckpt_width, opts.width_mult))
                n_copied, n_expanded, missing = expand_state_dict(model, ckpt["model"], old_width=ckpt_width, jitter=0.001)
                logger.info('Expand weights: %d layers copied exactly, %d layers channel-expanded.' % (n_copied, n_expanded))
                if missing:
                    logger.warning('Checkpoint keys not used: %s' % missing)
            elif abs(ckpt_width - opts.width_mult) < 1e-6:
                # 同宽度: 用 expand_state_dict 逐层拷贝, 容忍结构有变化的层(如 head 改版后
                # 新增的 BN / 已删除的大核 depthwise 卷积), 这些层保留随机初始化继续训练。
                n_copied, n_expanded, missing = expand_state_dict(model, ckpt["model"], old_width=ckpt_width, jitter=0.001)
                logger.info('Init weights: %d layers copied from checkpoint.' % n_copied)
                if missing:
                    logger.warning('Checkpoint keys not used (architecture changed, e.g. head): %s' % missing)
            else:
                # 变窄: Net2Net 缩权初始化。被裁掉的通道信息会丢失, 得到的只是较好的
                # 起点而非等价模型, 必须重新训练(不能 resume)。
                logger.info('Auto Net2Net-shrink init width_mult=%.2f -> %.2f.' % (ckpt_width, opts.width_mult))
                n_copied, n_shrunk, missing = shrink_state_dict(model, ckpt["model"], old_width=ckpt_width)
                logger.info('Shrink weights: %d layers copied exactly, %d layers channel-shrunk.' % (n_copied, n_shrunk))
                if missing:
                    logger.warning('Checkpoint keys not used (architecture changed, e.g. head): %s' % missing)
            logger.info('Weights initialized from %s; training from epoch 1 (optimizer/epoch not restored).' % ckpt_path)

        # 释放内存
        del ckpt
        logger.info('Load checkpoint completed.')

    # DP模式
    # if device.type != 'cpu' and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    # 加载数据
    train_dataset = LPRDataSet(args.train_img_dirs.split(","), opts.img_size, augment=True,
                               hard_list=opts.hard_list.split(",") if opts.hard_list else None,
                               hard_weight=opts.hard_weight)
    test_dataset = LPRDataSet(args.test_img_dirs.split(","), opts.img_size)
    if opts.repeat_weight > 0:
        log_repeat_stats(train_dataset, opts.repeat_weight)
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers,
                              pin_memory=cuda, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers,
                             pin_memory=cuda, collate_fn=test_dataset.collate_fn)

    # 缩权后各层激活分布已变, checkpoint 里的 BN running 统计不再匹配, 用训练数据重估一次
    if shrink_init and opts.bn_recalib_batches > 0:
        n_calib = recalibrate_bn(model, train_loader, device, max_batches=opts.bn_recalib_batches)
        logger.info('BatchNorm recalibrated on %d batches after shrink init.' % n_calib)

    # 设置已经进行的轮数
    scheduler.last_epoch = start_epoch - 2  # 因为 epoch 从 1 开始
    # 自动半精度优化
    scaler = torch.cuda.amp.GradScaler(enabled=cuda)

    best_acc = -1.0

    logger.info('Image sizes %d train, %d test' % (len(train_dataset), len(test_dataset)))
    logger.info('Using %d dataloader workers' % opts.workers)

    # 本次训练用的配置 + 每个 epoch 的结果都落在本次的 out_dir 下, 便于事后复现/画曲线
    cfg_path = save_run_config(opts.out_dir, opts, extra={
        'head_cfg': model.head_cfg,
        'time_steps': input_len,
        'n_params': sum(p.numel() for p in model.parameters()),
        'train_images': len(train_dataset),
        'test_images': len(test_dataset),
    })
    results_path = os.path.join(opts.out_dir, 'results.csv')
    run_start_time = time.time()
    logger.info('Run config saved to %s' % cfg_path)
    logger.info('Per-epoch results will be appended to %s' % results_path)

    logger.info('Starting training for %d epochs...' % start_epoch)
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        model.train()

        optimizer.zero_grad()

        mloss = .0
        mloss_opt = .0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train(%d/%d)' % (epoch, epochs))
        for i, batch in pbar:
            # 难例加权时数据集多返回一列样本权重(见 data/dataset.py 的 hard_list)
            if len(batch) == 4:
                imgs, labels, lengths, sample_w = batch
                sample_w = sample_w.to(device, non_blocking=True)
            else:
                imgs, labels, lengths = batch
                sample_w = None
            imgs, labels = imgs.to(device, non_blocking=True).float(), labels.to(device, non_blocking=True).float()

            # 准备 loss 计算的参数
            input_lengths, target_lengths = sparse_tuple_for_ctc(input_len, lengths)

            # Forward
            with torch.cuda.amp.autocast(enabled=cuda):
                x = model(imgs)
                x = x.permute(2, 0, 1)  # [batch_size, chars, width] -> [width, batch_size, chars]
                x = x.log_softmax(2).requires_grad_()
                # loss 口径: 未加权值用于日志(与改动前可比), 加权值用于反传
                loss, loss_log = compute_loss(ctc_loss, ctc_loss_none, x, labels,
                                              input_lengths, target_lengths, sample_w,
                                              opts.repeat_weight, device)

            # Backward
            scaler.scale(loss).backward()

            # 梯度裁剪(防发散)
            if opts.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts.grad_clip)

            # Optimize
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Print
            mloss = (mloss * i + loss_log.item()) / (i + 1)    # update mean losses (未加权)
            mloss_opt = (mloss_opt * i + loss.item()) / (i + 1)  # 实际反传的加权 loss
            lr = optimizer.param_groups[0]['lr']
            pbar.set_description('Train(%d/%d), lr: %.5f, mloss: %.5f' % (epoch, epochs, lr, mloss))

            # tb
            if epoch - start_epoch <= 3 and i < 3:
                if epoch == start_epoch and i == 0:
                    tb_writer.add_graph(model, imgs)  # add model to tensorboard

                f = os.path.join(opts.out_dir, 'train_batch_%d_%d.jpg' % (epoch, i))  # filename
                result = plot_images(images=imgs, fname=f)
                if result is not None:
                    tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)

            del x, loss

        # Scheduler
        scheduler.step()

        # Save model
        saved_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_type': 'adam' if opts.adam else 'sgd',
            'width_mult': opts.width_mult,
            'head_cfg': model.head_cfg
        }
        if (not opts.nosave or epoch == epochs) and epoch % opts.save_epochs == 0:
            torch.save(saved_data, os.path.join(opts.weights_dir, 'last.pt'))

        # Evaluate test
        test_mloss = test_macc = None
        rep_n = rep_correct = rep_deleted = 0
        rep_acc = None
        is_best = 0
        if (not opts.notest or epoch == epochs) and epoch % opts.test_epochs == 0:
            model.eval()
            test_mloss, test_macc, test_stats = test(model, test_loader, test_dataset, device,
                                                    ctc_loss, input_len, opts.float_test)

            rep_n = test_stats['repeat_total']
            rep_correct = test_stats['repeat_correct']
            rep_deleted = test_stats['repeat_deleted']
            rep_acc = float(test_stats['repeat_correct']) / rep_n if rep_n else 0.0
            logger.info('Repeat subset: acc=%.5f (n=%d, correct=%d, deleted=%d) | overall macc=%.5f'
                        % (rep_acc, rep_n, test_stats['repeat_correct'],
                           test_stats['repeat_deleted'], test_macc))

            # save best weights
            if best_acc <= test_macc:
                best_acc = test_macc
                is_best = 1

                if not opts.nosave:
                    torch.save(saved_data, os.path.join(opts.weights_dir, 'best.pt'))

            # tb
            tb_writer.add_scalar('val/mloss', test_mloss, epoch)
            tb_writer.add_scalar('val/macc', test_macc, epoch)
            tb_writer.add_scalar('val/macc_repeat', rep_acc, epoch)

        del saved_data

        # tb
        tb_writer.add_scalar('train/mloss', mloss, epoch)
        tb_writer.add_scalar('train/mloss_weighted', mloss_opt, epoch)
        tb_writer.add_scalar('train/lr', lr, epoch)

        # 每个 epoch 的结果追加到 results.csv(未做验证的 epoch, 验证列留空)
        append_results_csv(results_path, {
            'epoch': epoch,
            'lr': round(float(lr), 8),
            'train_loss': round(float(mloss), 6),
            'train_loss_weighted': round(float(mloss_opt), 6),
            'val_loss': '' if test_mloss is None else round(float(test_mloss), 6),
            'macc': '' if test_macc is None else round(float(test_macc), 6),
            'repeat_n': rep_n,
            'repeat_correct': rep_correct,
            'repeat_deleted': rep_deleted,
            'best_acc': '' if best_acc < 0 else round(float(best_acc), 6),
            'is_best': is_best,
            'epoch_time_s': round(time.time() - epoch_start_time, 2),
            'total_time_s': round(time.time() - run_start_time, 2),
        })

        # Split line
        logger.info('')

    # Save final weights
    torch.save({
        "epoch": epochs,
        "model": model.state_dict(),
        "width_mult": opts.width_mult,
        "head_cfg": model.head_cfg
    }, os.path.join(opts.weights_dir, 'final.pt'))

    logger.info('Training complete, .')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPRNet Training')
    parser.add_argument('--weights', type=str, default='', help='pretrained weights as INIT: train from epoch 1 (auto Net2Net-expand when --width-mult is wider, Net2Net-shrink + BatchNorm recalibration when it is narrower).')
    parser.add_argument('--resume', type=str, default='', help='checkpoint path to CONTINUE training (restores epoch/optimizer, same width_mult required).')
    parser.add_argument('--train_img_dirs', default="", help='the train images path')
    parser.add_argument('--test_img_dirs', default="", help='the test images path')
    parser.add_argument('--device', type=str, default='6', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training.')
    parser.add_argument('--batch-size', type=int, default=128, help='train batch size.')
    parser.add_argument('--img-size', default=(128, 48), type=lambda s: tuple(map(int, s.split(','))),
                        help='the image size, e.g. 128,48')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--width-mult', type=float, default=1, help='LPRNetV2 channel width multiplier (1.5=about 2x params, 1.0=original).')
    parser.add_argument('--head-ch', type=int, default=None, help='head branch conv output channels (None=no conv, only pooling).')
    parser.add_argument('--head-ksize', type=int, default=1, choices=[1, 3, 5], help='head branch conv kernel: 1=pointwise, 3/5=learnable spatial (needs --head-ch).')
    parser.add_argument('--head-pool', default='avg', choices=['avg', 'max'], help='pooling used to downsample each branch (grid geometry is auto-derived).')
    parser.add_argument('--head-norm', default='bn', choices=['bn', 'rms', 'none'], help="branch norm: bn (deploy-safe, dataset-level), rms (original LPRNet per-image energy norm), none.")
    parser.add_argument('--grid-h', type=int, default=4, help='head grid height (time-axis is averaged over it).')
    parser.add_argument('--grid-w', type=int, default=27, help='head grid width = CTC time steps T. Encode a plate of length L with adjacent repeats R needs T>=L+R; raise it (128x48 max 27, 160x48 max 35) to give repeated characters room for a separating blank.')
    parser.add_argument('--lpr-max-len', type=int, default=None, help='consistency check only: must equal --grid-w if set (CTC input_lengths is derived from grid_w).')
    parser.add_argument('--bn-recalib-batches', type=int, default=20, help='batches used to re-estimate BatchNorm running stats after a Net2Net-shrink init (0 disables).')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer.')
    parser.add_argument('--lr', type=float, default=None, help='initial learning rate (default: 0.001; auto 0.0003 when expanding --weights init).')
    parser.add_argument('--grad-clip', type=float, default=10.0, help='max gradient norm for clipping, 0 disables.')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum/Adam beta1.')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='LPRNet optimizer weight decay.')
    parser.add_argument('--repeat-weight', type=float, default=0.0,
                        help='连号样本加权系数 α: 样本权重 = 1 + α * 相邻重复字符数(>0 才启用). '
                             '只改训练 loss 的加权, 推理开销为 0; 建议 0.5~2 之间试.')
    parser.add_argument('--hard-list', type=str, default='',
                        help='难例清单(逗号分隔的 txt, 行格式同数据集列表 "<img_path> <plate> [type]"), '
                             '命中的图片在训练时额外加权(可用 test.py --dump-errors 生成).')
    parser.add_argument('--hard-weight', type=float, default=2.0, help='难例清单里样本的权重倍数(默认 2.0).')
    parser.add_argument('--workers', type=int, default=32, help='maximum number of dataloader workers.')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training.')
    parser.add_argument('--save-epochs', type=int, default=1, help='number of save interval epochs.')
    parser.add_argument('--test-epochs', type=int, default=1, help='number of test interval epochs.')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint.')
    parser.add_argument('--notest', action='store_true', help='only test final epoch.')
    parser.add_argument('--float-test', action='store_true', help='use float model run test.')
    parser.add_argument('--worker-dir', type=str, default='runs', help='worker dir.')
    parser.add_argument('--onnx', default=False, type=bool, help='show test image and its predict result or not.')
    args = parser.parse_args()

    # # 自动调整的参数
    # if args.workers < 0:
    #     if args.cache_images:
    #         args.workers = 1
    #     else:
    #         args.workers = os.cpu_count()
    # args.workers = min(os.cpu_count(), args.workers)

    # 打印参数
    logger.info("args: %s" % args)

    # 自动调整的参数(不打印)
    args.cache_dir = os.path.join(args.worker_dir, 'cache')
    args.out_dir = increment_dir(Path(args.worker_dir) / 'exp')
    args.weights_dir = os.path.join(args.out_dir, 'weights')

    # 参数处理后的初始化工作
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)

    # 控制台日志同时落盘到本次运行目录(与 args.json / results.csv 放一起, 便于事后排查)
    file_handler = logging.FileHandler(os.path.join(args.out_dir, 'train.log'), encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logger.info("Logging results to %s" % args.out_dir)

    # 开始训练
    main(args)
