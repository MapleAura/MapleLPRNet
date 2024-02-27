#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import math
import os
from pathlib import Path
from data.dataset import LPRDataSet
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.lprnet import LPRNetV2, CHARS
from test import test
from utils.general import increment_dir, plot_images, model_info, set_logging, \
    sparse_tuple_for_ctc, select_device

logger = logging.getLogger(__name__)
set_logging()


def main(opts):
    epochs = opts.epochs

    # 选择设备
    device = select_device(opts.device, opts.batch_size)
    cuda = device.type != 'cpu'
    logger.info('Use device %s.' % device)

    # 定义网络
    model = LPRNetV2(8, True, class_num=len(CHARS), dropout_rate=opts.dropout_rate).to(device)
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

    # lr 自动调整器
    lf = lambda e: (((1 + math.cos(e * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    del lf

    # TB
    logger.info('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % opts.worker_dir)
    tb_writer = SummaryWriter(log_dir=opts.out_dir)  # runs/exp0

    # Resume
    start_epoch = 1
    if opts.weights:
        ckpt = torch.load(opts.weights, map_location=device)

        # 加载网络
        model.load_state_dict(ckpt["model"])

        # 优化器
        if 'optimizer_type' in ckpt:  # 兼容 final.pt
            optimizer_type = 'adam' if opts.adam else 'sgd'
            if optimizer_type == ckpt['optimizer_type']:
                optimizer.load_state_dict(ckpt['optimizer'])
            else:
                logger.warning('Optimizer is changed, state has been lost.')

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (opts.weights, start_epoch - 1, start_epoch + epochs))
            epochs += start_epoch

        # 释放内存
        del ckpt, optimizer_type

        # Print
        logger.info('Load checkpoint completed.')

    # DP模式
    # if device.type != 'cpu' and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    # 加载数据
    train_dataset = LPRDataSet(args.train_img_dirs.split(","), opts.img_size)
    test_dataset = LPRDataSet(args.test_img_dirs.split(","), opts.img_size)
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers,
                              pin_memory=cuda, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers,
                             pin_memory=cuda, collate_fn=test_dataset.collate_fn)

    # 设置已经进行的轮数
    scheduler.last_epoch = start_epoch - 2  # 因为 epoch 从 1 开始
    # 自动半精度优化
    scaler = torch.cuda.amp.GradScaler(enabled=cuda)

    best_acc = -1.0

    logger.info('Image sizes %d train, %d test' % (len(train_dataset), len(test_dataset)))
    logger.info('Using %d dataloader workers' % opts.workers)
    logger.info('Starting training for %d epochs...' % start_epoch)
    for epoch in range(start_epoch, epochs + 1):
        model.train()

        optimizer.zero_grad()

        mloss = .0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train(%d/%d)' % (epoch, epochs))
        for i, (imgs, labels, lengths) in pbar:
            imgs, labels = imgs.to(device, non_blocking=True).float(), labels.to(device, non_blocking=True).float()

            # 准备 loss 计算的参数
            input_lengths, target_lengths = sparse_tuple_for_ctc(opts.lpr_max_len, lengths)

            # Forward
            with torch.cuda.amp.autocast(enabled=cuda):
                x = model(imgs)
                x = x.permute(2, 0, 1)  # [batch_size, chars, width] -> [width, batch_size, chars]
                x = x.log_softmax(2).requires_grad_()
                loss = ctc_loss(x, labels, input_lengths=input_lengths, target_lengths=target_lengths)

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Print
            mloss = (mloss * i + loss.item()) / (i + 1)  # update mean losses
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
            'optimizer_type': 'adam' if opts.adam else 'sgd'
        }
        if (not opts.nosave or epoch == epochs) and epoch % opts.save_epochs == 0:
            torch.save(saved_data, os.path.join(opts.weights_dir, 'last.pt'))

        # Evaluate test
        if (not opts.notest or epoch == epochs) and epoch % opts.test_epochs == 0:
            model.eval()
            test_mloss, test_macc = test(model, test_loader, test_dataset, device, ctc_loss, opts.lpr_max_len, opts.float_test)

            # save best weights
            if best_acc <= test_macc:
                best_acc = test_macc

                if not opts.nosave:
                    torch.save(saved_data, os.path.join(opts.weights_dir, 'best.pt'))

            # tb
            tb_writer.add_scalar('val/mloss', test_mloss, epoch)
            tb_writer.add_scalar('val/macc', test_macc, epoch)

        del saved_data

        # tb
        tb_writer.add_scalar('train/mloss', mloss, epoch)
        tb_writer.add_scalar('train/lr', lr, epoch)

        # Split line
        logger.info('')

    # Save final weights
    torch.save({
        "epoch": epochs,
        "model": model.state_dict()
    }, os.path.join(opts.weights_dir, 'final.pt'))

    logger.info('Training complete, .')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPRNet Training')
    parser.add_argument('--weights', type=str, default='', help='initial weights path.')
    parser.add_argument('--train_img_dirs', default="/mnt/code/0.dataset/CBLPRD330K/train.txt,/mnt/code/0.dataset/plate/train.txt", help='the train images path')
    parser.add_argument('--test_img_dirs', default="/mnt/code/0.dataset/plate/test.txt", help='the test images path')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training.')
    parser.add_argument('--batch-size', type=int, default=128, help='train batch size.')
    parser.add_argument('--img-size', default=(94, 24), help='the image size')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--lpr-max-len', default=18, help='license plate number max length.')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer.')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum/Adam beta1.')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='LPRNet optimizer weight decay.')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers.')
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

    logger.info("Logging results to %s" % args.out_dir)

    # 开始训练
    main(args)
