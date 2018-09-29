#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import pickle
import numpy as np
import sys
import logging

from backbone import EmbedNetwork
from datasets.Market1501 import Market1501


def eval():
    ## logging
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    ## restore model
    logger.info('restoring model')
    model = EmbedNetwork().cuda()
    model.load_state_dict(torch.load('./res/model.pkl'))
    model = nn.DataParallel(model)
    model.eval()

    ## load gallery dataset
    batchsize = 32
    ds = Market1501('datasets/Market-1501-v15.09.15/bounding_box_test', mode = 'gallery')
    dl = DataLoader(ds, batch_size = batchsize, num_workers = 4)


if __name__ == '__main__':
    pass
