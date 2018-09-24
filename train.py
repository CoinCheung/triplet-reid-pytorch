#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import sys
import os
import logging

from backbones import EmbedNetwork
from loss import TripletLoss
from selector import BatchHardTripletSelector
from batch_sampler import RegularBatchSampler
from datasets.Market1501 import Market1501
from optimizer import AdamOptimWrapper



def train():
    ## logging
    FORMAT = '%(levelname)s %(filename)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    ## model and loss
    net = EmbedNetwork().cuda()
    net = nn.DataParallel(net)
    triplet_loss = TripletLoss(margin = None).cuda() # TODO: add a margin for this

    ## optimizer
    optim = AdamOptimWrapper(net.parameters(), lr = 3e-4, t0 = 15000, t1 = 25000)


    ## dataloader
    ds = Market1501('datasets/Market-1501-v15.09.15/bounding_box_train', mode = 'train')
    sampler = RegularBatchSampler(ds, 18, 4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)
    selector = BatchHardTripletSelector()

    ## train
    count = 0
    while True:
        for it, (imgs, lbs) in enumerate(dl):
            net.train()
            imgs = imgs.cuda()
            lbs = lbs.cuda()
            embds = net(imgs)
            anchor, positives, negatives = selector(embds, lbs)

            loss = triplet_loss(anchor, positives, negatives)
            if it % 20 == 0 and it != 0:
                print(it)
                print(loss.detach().cpu().numpy())
            optim.zero_grad()
            loss.backward()
            optim.step()

            count += 1
            if count == 25000: break
        if count == 25000: break


    ## dump model
    if not os.path.exists('./res'): os.makedirs('./res')
    logger.info('saving trained model')
    torch.save(net.module.state_dict(), './res/model.pkl')



if __name__ == '__main__':
    train()
