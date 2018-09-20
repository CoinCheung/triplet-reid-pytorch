#!/usr/bin/python
# -*- encoding: utf-8 -*-



import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import cv2
import numpy as np


class TripletLoader():
    pass


class BatchHardSelector(BatchSampler):
    def __init__(self, dataset, n_class, n_num, *args, **kwargs):
        super(BatchHardSelector, self).__init(*args, **kwargs)
        self.n_class = n_class
        self.n_num = n_num
        self.batch_size = n_class * n_num
        self.lb_img_dict = dataset.lb_img_dict
        for k, v in self.lb_img_dict.items():
            random.shuffle(self.lb_img_dict[k])

    def __iter__(self):
        pass

    def __len__(self):
        return len(dataset) // self.batch_size
