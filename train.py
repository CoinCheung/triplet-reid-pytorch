#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import sys
import os
import logging
import time

from backbone import EmbedNetwork
from loss import TripletLoss
from triplet_selector import BatchHardTripletSelector
from batch_sampler import BatchSampler
from datasets.Market1501 import Market1501
from optimizer import AdamOptimWrapper
from logger import logger


torch.multiprocessing.set_sharing_strategy('file_system')

## TODO:
## 1. see training scheme of 300 epochs (change lr each epoch instead of iter)
## 2. see embedding length of 2048


def train():

    ## model and loss
    logger.info('setting up backbone model and loss')
    net = EmbedNetwork().cuda()
    net = nn.DataParallel(net)
    triplet_loss = TripletLoss(margin = None).cuda() # no margin means soft-margin

    ## optimizer
    optim = AdamOptimWrapper(net.parameters(), lr = 3e-4, wd = 5e-4, t0 = 15000, t1 = 25000)

    ## dataloader
    ds = Market1501('datasets/Market-1501-v15.09.15/bounding_box_train', mode = 'train')
    sampler = BatchSampler(ds, 18, 4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)
    selector = BatchHardTripletSelector()

    ## train
    times = []
    count = 0
    while True:
        ## TODO:
        ## 1. use infinite dataloader, dont forget to shuffle
        for it, (imgs, lbs, _) in enumerate(dl):
            st = time.time()
            net.train()
            imgs = imgs.cuda()
            lbs = lbs.cuda()
            embds = net(imgs)
            anchor, positives, negatives = selector(embds, lbs)

            loss = triplet_loss(anchor, positives, negatives)
            optim.zero_grad()
            loss.backward()
            optim.step()

            times.append(time.time() - st)
            if count % 20 == 0 and it != 0:
                loss_val = loss.detach().cpu().numpy()
                time_interval = sum(times) / len(times)
                times = []
                logger.info('iter: {}, loss: {:4f}, lr: {:4f}, time_avg: {:3f}'.format(count, loss_val, optim.lr, time_interval))

            count += 1
            if count == 25000: break
        if count == 25000: break

    ## dump model
    if not os.path.exists('./res'): os.makedirs('./res')
    logger.info('saving trained model')
    torch.save(net.module.state_dict(), './res/model.pkl')

    logger.info('everything finished')



if __name__ == '__main__':
    train()
