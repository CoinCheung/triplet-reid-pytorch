#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np


class Market1501(Dataset):
    '''
    a wrapper of Market1501 dataset
    '''
    def __init__(self, data_path, mode = 'train', *args, **kwargs):
        super(Market1501, self).__init__(*args, **kwargs)
        self.mode = mode
        self.data_path = data_path
        self.imgs = os.listdir(data_path)
        self.imgs = [el for el in self.imgs if el[-4:] == '.jpg']
        self.labels = [int(el.split('_')[0]) - 1 for el in self.imgs]
        self.imgs = [os.path.join(data_path, el) for el in self.imgs]
        self.lb_img_dict = dict()

        # useful for sampler
        lb_array = np.array(self.labels)
        for lb in self.labels:
            idx = lb_array[lb_array == lb]
            im_list = [self.imgs[i] for i in idx]
            self.lb_img_dict.update({lb: im_list})

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        return img, self.labels[idx]



if __name__ == "__main__":
    ds = Market1501('./Market-1501-v15.09.15/bounding_box_train', mode = 'train')
    im, lb = ds[14]
    #  cv2.imshow('img', im)
    #  cv2.waitKey(0)
    i = 0
    for k, v in ds.lb_img_dict.items():
        print(k, v)
        i += 1
        if i == 10: break

