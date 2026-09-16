#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from model.stnet import STNet


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)
    
class CBR(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(CBR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 2), padding=0),
            nn.BatchNorm2d(num_features=ch_out),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.block(x)
    
class RMSNorm2d(nn.Module):
    """原 LPRNet head 的归一化: 每个样本在 (C, H, W) 上除以能量 mean(f^2).

    它把各分支、各图片的响应尺度拉到同一量级(逐图对比度不变性), 是 head 精度的一部分.
    部署提示: 导出为 Pow / ReduceMean / Div 等数据相关缩放算子, 部分 NPU/量化工具链
    不支持; 目标芯片不支持时改用 head_norm='bn' (只做数据集级逐通道归一化).
    """

    def __init__(self, eps=1e-6):
        super(RMSNorm2d, self).__init__()
        self.eps = eps

    def forward(self, x):
        f_mean = torch.mean(torch.pow(x, 2), dim=(1, 2, 3), keepdim=True)
        return torch.div(x, f_mean + self.eps)


class LPRNetV2(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate, width_mult=1.0,
                 img_size=(128, 48), grid_size=(4, 18), head_ch=None, head_pool='avg',
                 head_norm='bn', head_ksize=1):
        super(LPRNetV2, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.width_mult = width_mult
        self.grid_h, self.grid_w = grid_size
        c1 = int(round(64 * width_mult))
        c2 = int(round(128 * width_mult))
        c3 = int(round(256 * width_mult))
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=c1, kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=c1),
            nn.ReLU(),  # 2
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
            #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            small_basic_block(ch_in=c1, ch_out=c2),    # *** 4 ***
            nn.BatchNorm2d(num_features=c2),
            nn.ReLU(),  # 6
            #nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            CBR(c2, c1),
            small_basic_block(ch_in=c1, ch_out=c3),   # 8
            nn.BatchNorm2d(num_features=c3),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=c3, ch_out=c3),   # *** 11 ***
            nn.BatchNorm2d(num_features=c3),   # 12
            nn.ReLU(),
            #nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            CBR(c3, c1),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=c1, out_channels=c3, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=c3),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=c3, out_channels=class_num, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        # 4 个多尺度特征点 -> 固定核/步长的池化下采样到统一网格 (grid_h, grid_w)
        # -> [1x1 卷积] -> 归一化 -> concat -> 高度平均 -> 1x1 卷积分类。
        # 池化核/步长由 img_size(宽,高)下的实际特征尺寸自动计算, 保证输出恰为
        # (grid_h, grid_w) 且相邻窗口相接/重叠(不丢失特征图边缘), 所以 128x48、
        # 146x48、160x48 等输入均支持; 换输入尺寸需重新实例化模型。
        # 注意: 旧版 head 的大核 depthwise 卷积(如 11x22 步长 11x6)训练后 99% 能量
        # 都在"去直流"分量上(逐通道 |sum w|/sum|w| 仅 0.05~0.10), 即它给每个通道学了
        # 一个不同的 ± 高通模板; 直接把 head 换成平均池化会把这种"逐通道不同模式"
        # 压成同一个窗口均值, 精度会掉。因此这里保留可学习空间卷积(见 head_ch/ksize):
        # 池化前先用 ksize x ksize 卷积给各通道提取不同局部模式, 再聚合。
        # head_ch=None: 各分支不做卷积, 只做池化(最省, 精度略低);
        # head_ch=int : 各分支先做 ksize x ksize 卷积压缩到 head_ch 通道。
        # head_ksize  : 1=逐点(最省), 3/5=局部空间卷积(更有表达力, 代价是该分支
        #               全分辨率上的 ch*head_ch*ksize^2 FLOPs)。
        # head_norm='bn'  : 逐分支 BN, 只有通用算子(池化/BN/Concat/1x1 卷积),
        #                   量化与 NPU 部署最稳, 但没有逐图对比度不变性;
        # head_norm='rms' : 原版 LPRNet 的 f/mean(f^2) 逐样本能量归一化, 精度与旧
        #                   head 对齐, 但含数据相关缩放, 部分芯片/量化工具链不支持;
        # head_norm='none': 不做归一化, 完全交给 container 卷积。
        self.keep_idx = (2, 6, 13, 22)
        assert head_pool in ('avg', 'max'), "head_pool must be 'avg' or 'max'"
        assert head_norm in ('bn', 'rms', 'none'), "head_norm must be 'bn', 'rms' or 'none'"
        assert head_ksize in (1, 3, 5) and head_ksize % 2 == 1, 'head_ksize must be 1, 3 or 5'
        assert head_ch or head_ksize == 1, 'head_ksize > 1 requires head_ch (set --head-ch)'
        pool_cls = nn.MaxPool2d if head_pool == 'max' else nn.AvgPool2d
        heads = []
        head_chs = []
        with torch.no_grad():
            x = torch.zeros(1, 3, img_size[1], img_size[0])
            for i, layer in enumerate(self.backbone.children()):
                x = layer(x)
                if i in self.keep_idx:
                    ch, fh, fw = x.shape[1], x.shape[2], x.shape[3]
                    sh, sw = fh // self.grid_h, fw // self.grid_w
                    assert sh >= 1 and sw >= 1 and fh >= self.grid_h and fw >= self.grid_w, \
                        'feature %dx%d too small for grid (%d,%d) at img_size %s' % (
                            fw, fh, self.grid_h, self.grid_w, (img_size[0], img_size[1]))
                    kh, kw = fh - (self.grid_h - 1) * sh, fw - (self.grid_w - 1) * sw
                    layers = []
                    if head_ch:
                        # 可学习空间投影: 先在全分辨率上做 ksize x ksize 卷积, 让各通道
                        # 各自提取不同的局部模式(旧 head 的 11x22 大核 depthwise 正是这个作用),
                        # 再池化聚合。ksize=1 时退化为纯通道混合。
                        layers += [nn.Conv2d(ch, head_ch, kernel_size=head_ksize,
                                             stride=1, padding=head_ksize // 2, bias=False),
                                   nn.BatchNorm2d(num_features=head_ch),
                                   nn.ReLU()]
                        out_ch = head_ch
                    else:
                        out_ch = ch
                    layers.append(pool_cls(kernel_size=(kh, kw), stride=(sh, sw), ceil_mode=False))
                    if head_norm == 'bn':
                        layers.append(nn.BatchNorm2d(num_features=out_ch))
                    elif head_norm == 'rms':
                        layers.append(RMSNorm2d())
                    head_chs.append(out_ch)
                    heads.append(nn.Sequential(*layers))
        self.heads = nn.ModuleList(heads)
        self.head_group_chs = tuple(head_chs)  # head 各分支输出通道, 供权重迁移判断布局
        # 网格高度方向的平均(等价于原来的 torch.mean(x, dim=2), 但导出为普通池化)
        self.grid_pool = nn.AvgPool2d(kernel_size=(self.grid_h, 1), stride=(self.grid_h, 1),
                                      ceil_mode=False)
        # head 结构参数, 训练时随 checkpoint 保存, 便于 test/export/detect 还原
        self.head_cfg = dict(grid_size=(self.grid_h, self.grid_w), head_ch=head_ch,
                             head_pool=head_pool, head_norm=head_norm, head_ksize=head_ksize)
        # 分类头: 各分支 concat 后 1x1 卷积映射到字符类别
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=sum(head_chs), out_channels=self.class_num,
                      kernel_size=(1, 1), stride=(1, 1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in self.keep_idx:  # (2, 6, 13, 22)
                keep_features.append(x)

        global_context = list()
        for f, head in zip(keep_features, self.heads):
            f = head(f)  # 池化下采样 -> (grid_h, grid_w)
            f = self.grid_pool(f)  # 网格高度平均 -> (1, grid_w)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)  # (B, class_num, 1, grid_w)
        logits = x.squeeze(2)  # (B, class_num, grid_w)

        return logits


class LPRNet(nn.Module):
    def __init__(self, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.class_num = class_num
        self.dropout_rate = dropout_rate
        self.stn = STNet()
        self.stage1 = nn.Sequential(  # 3x48x96
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # 64x48x96
            nn.MaxPool2d((2, 2), stride=(2, 2), ceil_mode=True),  # 64x24x48
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, groups=64),  # 64x24x48
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=(2, 0), groups=64), # 64x24x44  S
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            small_basic_block(ch_in=64, ch_out=128),  # 128x24x44
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self.outconv1 = nn.Sequential(  # 128x24x44 k(5,5) s(5,2)   Stage1
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=(5, 2), groups=128), # 128x4x20  k(3,3) s(1,1)
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self.down1 = nn.Sequential(
            small_basic_block(ch_in=128, ch_out=128),  # 128x24x44
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)  # 64x24x44
        )
        self.stage2 = nn.Sequential(  # 64x24x44  k(5,5) s(2,2) p(2,2)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2, groups=64),  # 64x12x22  S
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            small_basic_block(ch_in=64, ch_out=128),  # 128x12x22 S
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self.outconv2 = nn.Sequential(  # 128x12x22 k(5,3) s(2,1)   Stage2
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 3), stride=(2, 1), groups=128), # 128x4x20  k(3,3) s(1,1)
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            small_basic_block(ch_in=128, ch_out=128),  # 128x12x22
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)  # 128x12x22
        )
        self.stage3 = nn.Sequential(  # 128x12x22
            small_basic_block(ch_in=128, ch_out=128),  # 128x12x22 k(1,1) s(1,1)
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1),  # 256x12x22
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 13), stride=(2, 1), padding=(1, 6), groups=256),  # 256x6x22  k(3,3) s(1,1)
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, groups=256), # 256x4x20  k(3,3) s(1,1)
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, groups=512),  # 512x2x18
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.container2 = nn.Conv2d(in_channels=512, out_channels=class_num, kernel_size=(1, 17), stride=(1, 2), padding=(0, 8))

    def forward(self, x):
        x = self.stn(x)
        out1 = self.stage1(x)  # 128x24x44
        out = self.down1(out1)  # 64x24x44
        out2 = self.stage2(out)  # 128x12x22
        out = self.down2(out2)  # 128x12x22
        out3 = self.stage3(out)  # 256x4x20
        out1 = self.outconv1(out1)  # 128x4x20
        out2 = self.outconv2(out2)  # 128x4x20
        logits = torch.cat((out1, out2, out3), 1)  # 512x4x20
        logits = self.container(logits)  # 512x2x18
        top, bottom = torch.split(logits, 1, dim=2)
        logits = torch.cat((top, bottom), 3)  # 512x1x36
        logits = self.container2(logits)  # cx1x18
        logits = logits.permute(0, 1, 3, 2)  # 18xcx1
        logits = torch.Tensor.squeeze(logits, dim=3)  # nx18xc

        return logits

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新', '学', '港', '澳', '警', '使', '领', '应', '急', '挂',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', '-'
        ]
