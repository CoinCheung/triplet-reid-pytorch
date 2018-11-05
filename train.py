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
    selector = BatchHardTripletSelector()
    ds = Market1501('datasets/Market-1501-v15.09.15/bounding_box_train', is_train = True)
    sampler = BatchSampler(ds, 18, 4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)
    diter = iter(dl)

    ## train
    loss_avg = []
    count = 0
    t_start = time.time()
    while True:
        try:
            imgs, lbs, _ = diter.next()
        except StopIteration:
            diter = iter(dl)
            imgs, lbs, _ = diter.next()

        net.train()
        imgs = imgs.cuda()
        lbs = lbs.cuda()
        embds = net(imgs)
        anchor, positives, negatives = selector(embds, lbs)

        loss = triplet_loss(anchor, positives, negatives)
        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_avg.append(loss.detach().cpu().numpy())
        if count % 20 == 0 and count != 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            t_end = time.time()
            time_interval = t_end - t_start
            logger.info('iter: {}, loss: {:4f}, lr: {:4f}, time_avg: {:3f}'.format(count, loss_avg, optim.lr, time_interval))
            loss_avg = []
            t_start = t_end

        count += 1
        if count == 25000: break

    ## dump model
    if not os.path.exists('./res'): os.makedirs('./res')
    logger.info('saving trained model')
    torch.save(net.module.state_dict(), './res/model.pkl')

    logger.info('everything finished')



if __name__ == '__main__':
    train()
