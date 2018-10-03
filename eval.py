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
import argparse

from utils import pdist_np as pdist

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--gallery_embs',
            dest = 'gallery_embs',
            type = str,
            required = True,
            help = 'path to embeddings of gallery dataset'
            )
    parse.add_argument(
            '--query_embs',
            dest = 'query_embs',
            type = str,
            required = True,
            help = 'path to embeddings of query dataset'
            )

    return parse.parse_args()


def evaluate(args):
    ## logging
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    ## load embeddings
    logger.info('loading gallery embeddings')
    with open(args.gallery_embs, 'rb') as fr:
        gallery_dict = pickle.load(fr)
        emb_gallery, lb_gallery = gallery_dict['embeddings'], gallery_dict['labels']
    print(emb_gallery.shape)
    print(lb_gallery.shape)
    logger.info('loading query embeddings')
    with open(args.query_embs, 'rb') as fr:
        query_dict = pickle.load(fr)
        emb_query, lb_query = query_dict['embeddings'], query_dict['labels']
    print(emb_query.shape)
    print(lb_query.shape)

    dist_mtx = pdist(emb_query, emb_gallery)
    print(dist_mtx.shape)

    ## compute mAP
    indices = np.argsort(dist_mtx, axis = 1)



if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
