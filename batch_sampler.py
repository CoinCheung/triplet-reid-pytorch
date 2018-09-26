#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import cv2
import numpy as np
import random



class RegularBatchSampler(Sampler):
    '''
    sampler used in dataloader. method __iter__ should output the indices each time it is called
    '''
    def __init__(self, dataset, n_class, n_num, *args, **kwargs):
        super(RegularBatchSampler, self).__init__(dataset, *args, **kwargs)
        self.n_class = n_class
        self.n_num = n_num
        self.batch_size = n_class * n_num
        self.dataset = dataset
        self.labels = np.array(dataset.labels)
        self.len = len(dataset) // self.batch_size
        self.lb_img_dict = dataset.lb_img_dict
        for k, v in self.lb_img_dict.items():
            random.shuffle(self.lb_img_dict[k])

    def __iter__(self):
        count = 0
        while count <= self.len:
            label_batch = np.random.choice(self.labels, self.n_class, replace = False)
            idx = []
            for lb in label_batch:
                if len(self.lb_img_dict[lb]) > self.n_num:
                    idx_smp = np.random.choice(self.lb_img_dict[lb],
                            self.n_num, replace = False)
                else:
                    idx_smp = np.random.choice(self.lb_img_dict[lb],
                            self.n_num, replace = True)
                #  for i, im_idx in enumerate(idx_smp):
                #      im_name = self.dataset.imgs[im_idx]
                #      im = cv2.imread(im_name)
                #      cv2.imshow('img_{}'.format(i), im)
                #  cv2.waitKey(0)
                idx.extend(list(idx_smp))
            yield idx
            count += 1

    def __len__(self):
        return len(dataset) // self.batch_size


if __name__ == "__main__":
    from datasets.Market1501 import Market1501
    ds = Market1501('datasets/Market-1501-v15.09.15/bounding_box_train', mode = 'train')
    sampler = RegularBatchSampler(ds, 5, 4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)

    for i, (ims, lbs) in enumerate(dl):
        print(ims.shape)
        print(lbs.shape)
        #  if i == 4: break

