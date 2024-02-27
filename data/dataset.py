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
class LPRDataSet(torch.utils.data.Dataset):
    def __init__(self, data_set, img_size):

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

    def load_img(self, idx):
        filename = self.img_paths[idx]
        labels = filename.split("#")[1]
        name = filename.split("#")[0]
        image = cv2.imread(name)
        height, width, _ = image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            image = cv2.resize(image, self.img_size)

        # 归一化
        image = (image.astype('float32') - 127.5) * 0.007843

        # to tensor
        image = torch.from_numpy(image.transpose((2, 0, 1))).contiguous()

        if random.random() > .5:
            image = -image

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

        return image, label, len(label)

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def collate_fn(batch):
        images, labels, lengths = zip(*batch)
        labels = functools.reduce(operator.concat, labels)

        return torch.stack(images, 0), torch.tensor(labels), lengths
