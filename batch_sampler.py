#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import cv2
import numpy as np
import random
import logging
import sys



class BatchSampler(Sampler):
    '''
    sampler used in dataloader. method __iter__ should output the indices each time it is called
    '''
    def __init__(self, dataset, n_class, n_num, *args, **kwargs):
        super(BatchSampler, self).__init__(dataset, *args, **kwargs)
        self.n_class = n_class
        self.n_num = n_num
        self.batch_size = n_class * n_num
        self.dataset = dataset
        self.labels = np.array(dataset.lb_ids)
        self.labels_uniq = np.array(list(dataset.lb_ids_uniq))
        self.len = len(dataset) // self.batch_size
        self.lb_img_dict = dataset.lb_img_dict

    def __iter__(self):
        count = 0
        np.random.shuffle(self.labels_uniq)
        for k, v in self.lb_img_dict.items():
            random.shuffle(self.lb_img_dict[k])
        while count <= self.len:
            label_batch = np.random.choice(self.labels_uniq, self.n_class, replace = False)
            idx = []
            for lb in label_batch:
                if len(self.lb_img_dict[lb]) > self.n_num:
                    idx_smp = np.random.choice(self.lb_img_dict[lb],
                            self.n_num, replace = False)
                else:
                    idx_smp = np.random.choice(self.lb_img_dict[lb],
                            self.n_num, replace = True)
                idx.extend(list(idx_smp))
            yield idx
            count += 1

    def __len__(self):
        return len(dataset) // self.batch_size


if __name__ == "__main__":
    from datasets.Market1501 import Market1501
    ds = Market1501('datasets/Market-1501-v15.09.15/bounding_box_train', mode = 'train')
    sampler = BatchSampler(ds, 5, 4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)

    for i, (ims, lbs) in enumerate(dl):
        print(ims.shape)
        print(lbs.shape)
        #  if i == 4: break

