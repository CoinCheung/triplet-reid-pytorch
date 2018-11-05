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
import cv2

from backbone import EmbedNetwork
from datasets.Market1501 import Market1501


torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--store_pth',
            dest = 'store_pth',
            type = str,
            required = True,
            help = 'path that the embeddings are stored: e.g.: ./res/emb.pkl',
            )
    parse.add_argument(
            '--data_pth',
            dest = 'data_pth',
            type = str,
            required = True,
            help = 'path that the raw images are stored',
            )

    return parse.parse_args()



def embed(args):
    ## logging
    FORMAT = '%(levelname)s %(filename)s:%(lineno)d: %(message)s'
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
    ds = Market1501(args.data_pth, is_train = False)
    dl = DataLoader(ds, batch_size = batchsize, drop_last = False, num_workers = 4)


    ## embedding samples
    logger.info('start embedding')
    all_iter_nums = len(ds) // batchsize + 1
    embeddings = []
    label_ids = []
    label_cams = []
    for it, (img, lb_id, lb_cam) in enumerate(dl):
        print('\r=======>  processing iter {} / {}'.format(it, all_iter_nums),
                end = '', flush = True)
        label_ids.append(lb_id)
        label_cams.append(lb_cam)
        embds = []
        for im in img:
            im = im.cuda()
            embd = model(im).detach().cpu().numpy()
            embds.append(embd)
        embed = sum(embds) / len(embds)
        embeddings.append(embed)
    print('  ...   completed')

    embeddings = np.vstack(embeddings)
    label_ids = np.hstack(label_ids)
    label_cams = np.hstack(label_cams)

    ## dump results
    logger.info('dump embeddings')
    embd_res = {'embeddings': embeddings, 'label_ids': label_ids, 'label_cams': label_cams}
    with open(args.store_pth, 'wb') as fw:
        pickle.dump(embd_res, fw)

    logger.info('embedding finished')


if __name__ == '__main__':
    args = parse_args()
    embed(args)
