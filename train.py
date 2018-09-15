#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision

import sys
import logging

from backbones import EmbedNetwork


def train():

    ## logging
    FORMAT = '%(levelname)s %(filename)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    ## model and loss
    net = Embednetwork()





if __name__ == '__main__':
    train()
