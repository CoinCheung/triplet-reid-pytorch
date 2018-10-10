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

## TODO: change args_parser back to required

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

    ## check scikit-learn version
    import sklearn
    assert sklearn.__version__ == '0.18.1', 'eval.py require scikit-learn version to be 0.18.1, but you got a version of {}'.format(sklearn.__version__)


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
    ### TODO: pick out useless code and comments
    dist_mtx = pdist(emb_query, emb_gallery)
    indices = np.argsort(dist_mtx, axis = 1)
    # find images in query set that have identical cam id and pid overlaps with gallery set (nxm matrix)
    lb_ids_matchs = lb_ids_query[:, np.newaxis] != lb_ids_gallery
    lb_cams_matchs = lb_cams_query[:, np.newaxis] != lb_cams_gallery
    query_ovlp = np.logical_or(lb_ids_matchs, lb_cams_matchs)
    # set gallery images whose pids are -1 to invalid
    n_qu, n_ga = dist_mtx.shape
    invalid_gallery = np.tile((lb_ids_gallery == -1), n_qu).reshape(n_qu, n_ga)
    #  invalid_mask = np.logical_or(invalid_gallery, query_ovlp)
    dist_mtx[invalid_gallery] = np.inf
    lb_ids_matchs[invalid_gallery] = False

    ## compute mAP
    # change distance into score
    matchs = lb_ids_gallery[indices] == lb_ids_query[:, np.newaxis]
    #  scores = 1.0 / (1 + dist_mtx)

    aps = []
    for i in range(n_qu):
        score = 1.0 / (1 + dist_mtx[i][indices[i]])
        if np.sum(query_ovlp[i]) == 0:
            logger.info('invalid query')
            continue
        ap = average_precision_score(matchs[i], score)
        if np.isnan(ap):
            logger.info('having an ap of Nan, neglecting')
            continue
        aps.append(ap)
    mAP = sum(aps) / len(aps)

    print("map is: {}".format(mAP))


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
