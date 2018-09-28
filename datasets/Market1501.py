#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os
import torch
import torchvision
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
        self.imgs = [el for el in self.imgs if el[-4:] == '.jpg']
        self.labels = [int(el.split('_')[0]) - 1 for el in self.imgs]
        self.imgs = [os.path.join(data_path, el) for el in self.imgs]
        self.lb_img_dict = dict()
        if self.mode == 'train':
            self.trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize((288, 144)),
                torchvision.transforms.RandomCrop((256, 128)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        elif self.mode == 'gallery':
            self.trans1 = torchvision.transforms.Compose([
                torchvision.transforms.Resize((288, 144)),
                torchvision.transforms.FiveCrop((256, 128)),
            ])
            self.trans2 = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(0),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            self.trans3 = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        # useful for sampler
        lb_array = np.array(self.labels)
        for lb in self.labels:
            idx = np.where(lb_array == lb)[0]
            self.lb_img_dict.update({lb: idx})

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        img = Image.fromarray(img, 'RGB')
        if self.mode == 'train':
            img = self.trans(img)
        elif self.mode == 'gallery':
            img_crops = self.trans1(img)
            img1 = torch.stack([self.trans2(crop) for crop in img_crops])
            img2 = torch.stack([self.trans3(crop) for crop in img_crops])
            img = torch.cat([img1, img2])
        return img, self.labels[idx]



if __name__ == "__main__":
    ds = Market1501('./Market-1501-v15.09.15/bounding_box_train', mode = 'gallery')
    im, lb = ds[14]
    print(im.shape)
    #  cv2.imshow('img', im)
    #  cv2.waitKey(0)
    i = 0
    for k, v in ds.lb_img_dict.items():
        #  print(k, v)
        i += 1
        if i == 10: break

