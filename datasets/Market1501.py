#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image




class Market1501(Dataset):
    '''
    a wrapper of Market1501 dataset
    '''
    def __init__(self, data_path, mode = 'train', *args, **kwargs):
        super(Market1501, self).__init__(*args, **kwargs)
        self.mode = mode
        self.data_path = data_path
        self.imgs = os.listdir(data_path)
        self.imgs = [el for el in self.imgs if os.path.splitext(el)[1] == '.jpg']
        self.lb_ids = [int(el.split('_')[0]) for el in self.imgs]
        self.lb_cams = [int(el.split('_')[1][1]) for el in self.imgs]
        self.imgs = [os.path.join(data_path, el) for el in self.imgs]
        if self.mode == 'train':
            self.trans = transforms.Compose([
                transforms.Resize((288, 144)),
                transforms.RandomCrop((256, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
            ])
        elif self.mode == 'gallery' or self.mode == 'query':
            self.trans_tuple = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
                ])
            self.Lambda = transforms.Lambda(
                lambda crops: [self.trans_tuple(crop) for crop in crops])
            self.trans = transforms.Compose([
                transforms.Resize((288, 144)),
                transforms.TenCrop((256, 128)),
                self.Lambda,
            ])
        else:
            raise ValueError('unsupport mode of {} for Market1501'. format(self.mode))

        # useful for sampler
        self.lb_img_dict = dict()
        self.lb_ids_uniq = set(self.lb_ids)
        lb_array = np.array(self.lb_ids)
        for lb in self.lb_ids_uniq:
            idx = np.where(lb_array == lb)[0]
            self.lb_img_dict.update({lb: idx})

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = self.trans(img)
        return img, self.lb_ids[idx], self.lb_cams[idx]



if __name__ == "__main__":
    ds = Market1501('./Market-1501-v15.09.15/bounding_box_train', mode = 'gallery')
    #  im, lb = ds[14]
    #  print(im.shape)
    #  #  cv2.imshow('img', im)
    #  #  cv2.waitKey(0)
    #  i = 0
    #  for k, v in ds.lb_img_dict.items():
    #      #  print(k, v)
    #      i += 1
    #      if i == 10: break

    im = cv2.imread('Market-1501-v15.09.15/query/0530_c3s1_149183_00.jpg')
    print(im.shape)
    ToTensor = torchvision.transforms.ToTensor()
    norm = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    deal = torchvision.transforms.Compose([ToTensor, norm])
    Lambda = torchvision.transforms.Lambda(
        lambda crops: torch.stack([deal(crop) for crop in crops]))
    im = Image.fromarray(im, 'RGB')
    crop1 = torchvision.transforms.Resize((288, 144))
    crop2 = torchvision.transforms.TenCrop((256, 128))
    trans = torchvision.transforms.Compose([
        crop1, crop2, Lambda
        ])
    im = trans(im)
    print(im.shape)
    print(len(im))
    print(im[0].shape)

    print(len(ds))
    print(ds[12935])
