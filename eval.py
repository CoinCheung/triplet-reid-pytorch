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

## TODO: change args_parser

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
        emb_query, lb_ids_query, lb_cams_query = query_dict['embeddings'], query_dict['label_ids'], query_dict['label_cams']

    ## compute and clean distance matrix
    dist_mtx = pdist(emb_query, emb_gallery)
    # find images in query set that have identical cam id and pid overlaps with gallery set (nxm matrix)
    lb_ids_matchs = lb_ids_query[:, np.newaxis] == lb_ids_gallery
    lb_cams_matchs = lb_cams_query[:, np.newaxis] == lb_cams_gallery
    query_ovlp = np.logical_and(lb_ids_matchs, lb_cams_matchs)
    # set query images whose pids are -1 to invalid
    n_qu, n_ga = dist_mtx.shape
    invalid_query = np.repeat((lb_ids_query == -1), n_ga, 0).reshape(n_qu, n_ga)
    invalid_mask = np.logical_or(invalid_query, query_ovlp)
    dist_mtx[invalid_mask] = np.inf
    lb_ids_matchs[invalid_mask] = False

    ## compute mAP
    # change distance into score
    scores = 1.0 / (1 + dist_mtx)

    aps = []
    for i in range(n_qu):
        ap = average_precision_score(lb_ids_matchs[i], scores[i])
        if np.isnan(ap):
            logger.info('having an ap of Nan, neglecting')
            continue
        aps.append(ap)
    mAP = sum(aps) / len(aps)

    print("map is: {}".format(mAP))
    print(max(aps))
    print(min(aps))



if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
