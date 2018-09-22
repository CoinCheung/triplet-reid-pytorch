#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision

import sys
import logging

from backbones import EmbedNetwork
from loss import TripletLoss
from selector import BatchHardTripletSelector
from batch_sampler import RegularBatchSampler
from datasets.Market1501 import Market1501
from torch.utils.data import DataLoader



def train():
    ## logging
    FORMAT = '%(levelname)s %(filename)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    ## model and loss
    net = EmbedNetwork().cuda()
    net = nn.DataParallel(net)
    triplet_loss = TripletLoss(margin = None) # TODO: add a margin for this

    ## optimizer

    ## dataloader
    ds = Market1501('datasets/Market-1501-v15.09.15/bounding_box_train', mode = 'train')
    sampler = RegularBatchSampler(ds, 18, 4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)
    selector = BatchHardTripletSelector()

    ## train
    for it, (imgs, lbs) in enumerate(dl):
        imgs = imgs.cuda()
        lbs = lbs.cuda()
        embds = net(imgs)
        print(embds.shape)
        anchor, positives, negatives = selector(embds, lbs)
        print(anchor.shape)
        print(positives.shape)
        print(negatives.shape)

        break




if __name__ == '__main__':
    train()
