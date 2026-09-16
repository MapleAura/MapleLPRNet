import logging
import random
import time
import operator
import functools
import cv2
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import os
from model.lprnet import CHARS

logger = logging.getLogger(__name__)
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

filter = ["新能源小型车", "普通蓝牌",  "黑色车牌"] #新能源大型车 单层黄牌


def color_augment(image):
    """训练时随机色彩增强: 亮度/对比度、饱和度/色相、通道白平衡偏移。推理不使用。"""
    if random.random() < 0.9:
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-25, 25)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if random.random() < 0.9:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-6, 6)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.6, 1.4), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.75, 1.25), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.6:
        gain = np.array([random.uniform(0.85, 1.15) for _ in range(3)], dtype=np.float32)
        image = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return image


class LPRDataSet(torch.utils.data.Dataset):
    def __init__(self, data_set, img_size, augment=False, hard_list=None, hard_weight=2.0):

        self.img_dir = data_set   
        self.img_paths = []
        for dir in data_set:
            self.parent = os.path.dirname(dir)
            with open(dir) as f:
                lines = f.readlines()
            for line in lines:
                line = line.split(" ")
                if line[2].strip() in filter:
                    self.img_paths += [self.parent + "/" + line[0] + "#" + line[1]]
        random.shuffle(self.img_paths)
        self.img_size = img_size
        self.augment = augment

        # 难例加权: hard_list 里出现过的图片在训练时给 hard_weight 倍样本权重。
        # 只影响 loss 的加权(见 train.py), 推理开销为 0。
        # 每行支持两种格式: "<img_path> <plate> [type]"(与数据集列表同格式)
        # 或 "<img_path>#<plate>"; 只按文件名匹配, 不要求路径前缀一致。
        self.sample_weights = None
        if hard_list:
            keys = set()
            for path in hard_list:
                if not path or not os.path.isfile(path):
                    logger.warning('hard_list not found, ignore: %s' % path)
                    continue
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        img = line.split(" ")[0].split("#")[0]
                        keys.add(os.path.basename(img))
            if keys:
                hit = 0
                self.sample_weights = []
                for p in self.img_paths:
                    if os.path.basename(p.split("#")[0]) in keys:
                        self.sample_weights.append(hard_weight)
                        hit += 1
                    else:
                        self.sample_weights.append(1.0)
                logger.info('Hard-example weighting: %d/%d samples x%.2f (from %s)'
                            % (hit, len(self.img_paths), hard_weight, ','.join(hard_list)))

    def load_img(self, idx):
        filename = self.img_paths[idx]
        labels = filename.split("#")[1]
        name = filename.split("#")[0]
        image = cv2.imread(name)
        height, width, _ = image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            image = cv2.resize(image, self.img_size)

        # 训练时色彩增强(推理时 augment=False 不执行)
        if self.augment:
            image = color_augment(image)

        # 归一化
        image = (image.astype('float32') - 127.5)  / 127.5

        # to tensor
        image = torch.from_numpy(image.transpose((2, 0, 1))).contiguous()

        # if random.random() > .5:
        #     image = -image

        label = []
        for c in labels:
            if c == 'I':
                c = '1'
            if c == 'O':
                c = '0'
            label.append(CHARS_DICT[c])

        return image, label

    def __getitem__(self, index):

        image, label = self.load_img(index)

        if self.sample_weights is not None:
            return image, label, len(label), self.sample_weights[index]
        return image, label, len(label)

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def collate_fn(batch):
        # 带样本权重(难例加权)时返回 4 元组, 否则保持原来的 3 元组
        if len(batch[0]) == 4:
            images, labels, lengths, weights = zip(*batch)
            labels = functools.reduce(operator.concat, labels)
            return torch.stack(images, 0), torch.tensor(labels), lengths, \
                torch.tensor(weights, dtype=torch.float32)
        images, labels, lengths = zip(*batch)
        labels = functools.reduce(operator.concat, labels)

        return torch.stack(images, 0), torch.tensor(labels), lengths
