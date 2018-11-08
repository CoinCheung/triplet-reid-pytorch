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
        self.iter_num = len(self.labels_uniq) // self.n_class

    def __iter__(self):
        curr_p = 0
        np.random.shuffle(self.labels_uniq)
        for k, v in self.lb_img_dict.items():
            random.shuffle(self.lb_img_dict[k])
        for i in range(self.iter_num):
            label_batch = self.labels_uniq[curr_p: curr_p + self.n_class]
            curr_p += self.n_class
            idx = []
            for lb in label_batch:
                if len(self.lb_img_dict[lb]) > self.n_num:
                    idx_smp = np.random.choice(self.lb_img_dict[lb],
                            self.n_num, replace = False)
                else:
                    idx_smp = np.random.choice(self.lb_img_dict[lb],
                            self.n_num, replace = True)
                idx.extend(idx_smp.tolist())
            yield idx

    def __len__(self):
        return self.iter_num


if __name__ == "__main__":
    from datasets.Market1501 import Market1501
    ds = Market1501('datasets/Market-1501-v15.09.15/bounding_box_train', is_train = False)
    sampler = BatchSampler(ds, 18, 4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)
    import itertools

    diter = itertools.cycle(dl)

    while True:
        ims, lbs, _ = next(diter)
        print(lbs.shape)
    print(len(list(ds.lb_ids_uniq)))
