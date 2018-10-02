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
        emb_gallery = pickle.load(fr)
    print(emb_gallery['embeddings'].shape)
    print(emb_gallery['labels'].shape)
    logger.info('loading query embeddings')
    with open(args.query_embs, 'rb') as fr:
        emb_query = pickle.load(fr)
    print(emb_query['embeddings'].shape)
    print(emb_query['labels'].shape)




if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
