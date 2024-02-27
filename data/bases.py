import torch
import os
from typing import List, Dict


class Img(object):
    def __init__(self, src: str, label: str = None, rect: [int] = None, color: str = None):
        """
        :param src: 图片路径
        :param label: 正确的车牌号
        :param color: 车牌颜色
        :param rect: [x, y, x, y], 用于标记车牌位置
        """
        self.src = src
        self.label = label
        self.color = color
        self.rect = rect


class ImageDataset(object):
    def __init__(self, source_dir, cache_dir, train=True):
        self.cache_dir = cache_dir
        self.im_files = []

        self.file_name = 'cache_{}.pt'.format(os.path.basename(source_dir).split(".")[0])

    def save_cache(self, data, key: str = None):
        path = os.path.join(self.cache_dir, self.file_name)
        if key:
            torch.save({key: data}, path)
        else:
            torch.save(data, path)

    def load_cache(self, key: str = None):
        path = os.path.join(self.cache_dir, self.file_name)
        if os.path.exists(path):
            cache = torch.load(path)
            if key:
                if key in cache:
                    return cache[key]
            return cache

    def __getitem__(self, index):
        return self.im_files[index]

    def __len__(self):
        return len(self.im_files)
