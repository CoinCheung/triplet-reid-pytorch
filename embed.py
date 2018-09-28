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



def embed():
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

    ## embedding samples
    logger.info('start embedding')
    all_iter_nums = len(ds) // batchsize
    embeddings = []
    labels = []
    for it, (img, lbs) in enumerate(dl):
        sys.stdout.write('\r=======>  processing iter {} / {}'.format(it, all_iter_nums))
        sys.stdout.flush()
        img = img.cuda()
        _, _, C, H, W = img.shape
        img = img.contiguous().view(-1, C, H, W)
        lbs = lbs.contiguous().detach().numpy()
        embd = model(img).detach().cpu().numpy()
        embeddings.append(embd)
        labels.append(lbs)
    print('  ...   completed')

    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)

    ## aggregate embeddings
    logger.info('aggregating embeddings')
    N, L = embeddings.shape
    embeddings = embeddings.reshape(int(N / 10), 10, L).mean(axis = 1)

    ## dump results
    logger.info('dump embeddings')
    embd_res = {'embeddings': embeddings, 'labels': labels}
    with open('res/embds.pkl', 'wb') as fw:
        pickle.dump(embd_res, fw)

    logger.info('everything finished')


if __name__ == '__main__':
    embed()
