#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch

import pickle
import numpy as np
import sys
import logging
import argparse
from sklearn.metrics import average_precision_score

from utils import pdist_np as pdist

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--gallery_embs',
            dest = 'gallery_embs',
            type = str,
            #  required = True,
            default = './res/emb_gallery.pkl',
            help = 'path to embeddings of gallery dataset'
            )
    parse.add_argument(
            '--query_embs',
            dest = 'query_embs',
            type = str,
            #  required = True,
            default = './res/emb_query.pkl',
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
        emb_gallery, lb_ids_gallery, lb_cams_gallery = gallery_dict['embeddings'], gallery_dict['label_ids'], gallery_dict['label_cams']
    logger.info('loading query embeddings')
    with open(args.query_embs, 'rb') as fr:
        query_dict = pickle.load(fr)
        emb_query, lb_ids_query, lb_cams_query = query_dict['embeddings'], gallery_dict['label_ids'], gallery_dict['label_cams']

    print(np.max(lb_ids_query))
    print(np.min(lb_ids_query))
    dist_mtx = pdist(emb_query, emb_gallery)

    ## compute mAP
    n_qu, n_ga = dist_mtx.shape
    indices = np.argsort(dist_mtx, axis = 1)
    #  print(indices.shape)
    correct = (lb_ids_gallery == lb_ids_query[:, np.newaxis])
    print(lb_ids_gallery[indices])
    print(correct.shape)
    aps = np.zeros(n_qu)
    query_valid = np.zeros(n_qu)
    for i in range(n_qu):
        valid = ((lb_ids_gallery[indices[i]]) != lb_ids_query[i] |
                (lb_cams_gallery[indices[i]] != lb_cams_query[i]))
        #  print(correct.shape)
        y_true = correct[i, valid]
        y_score = - dist_mtx[i][indices[i]][valid]
        if not np.any(y_true): continue
        query_valid[i] = 1
        aps[i] = average_precision_score(y_true, y_score)

    if len(aps) == 0:
        raise RuntimeError('No valid query')

    mAP = float(np.sum(aps)) / np.sum(query_valid)
    print("map is: {}".format(mAP))




if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
