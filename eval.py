#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch

import pickle
import numpy as np
import sys
import logging
import argparse
from tqdm import tqdm

from utils import pdist_np as pdist


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--gallery_embs',
            dest = 'gallery_embs',
            type = str,
            default = './res/emb_gallery.pkl',
            help = 'path to embeddings of gallery dataset'
            )
    parse.add_argument(
            '--query_embs',
            dest = 'query_embs',
            type = str,
            default = './res/emb_query.pkl',
            help = 'path to embeddings of query dataset'
            )
    parse.add_argument(
            '--cmc_rank',
            dest = 'cmc_rank',
            type = int,
            default = 1,
            help = 'path to embeddings of query dataset'
            )

    return parse.parse_args()


def evaluate(args):
    ## logging
    FORMAT = '%(levelname)s %(filename)s:%(lineno)d: %(message)s'
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
    n_q, n_g = dist_mtx.shape
    indices = np.argsort(dist_mtx, axis = 1)
    matches = lb_ids_gallery[indices] == lb_ids_query[:, np.newaxis]
    matches = matches.astype(np.int32)
    all_aps = []
    all_cmcs = []
    logger.info('starting evaluating ...')
    for qidx in tqdm(range(n_q)):
        qpid = lb_ids_query[qidx]
        qcam = lb_cams_query[qidx]

        order = indices[qidx]
        pid_diff = lb_ids_gallery[order] != qpid
        cam_diff = lb_cams_gallery[order] != qcam
        useful = lb_ids_gallery[order] != -1
        keep = np.logical_or(pid_diff, cam_diff)
        keep = np.logical_and(keep, useful)
        match = matches[qidx][keep]

        if not np.any(match): continue

        cmc = match.cumsum()
        cmc[cmc > 1] = 1
        all_cmcs.append(cmc[:args.cmc_rank])

        num_real = match.sum()
        match_cum = match.cumsum()
        match_cum = [el / (1.0 + i) for i, el in enumerate(match_cum)]
        match_cum = np.array(match_cum) * match
        ap = match_cum.sum() / num_real
        all_aps.append(ap)

    assert len(all_aps) > 0, "NO QUERY MATCHED"
    mAP = sum(all_aps) / len(all_aps)
    all_cmcs = np.array(all_cmcs, dtype = np.float32)
    cmc = np.mean(all_cmcs, axis = 0)

    print('mAP is: {}, cmc is: {}'.format(mAP, cmc))


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
