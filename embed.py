#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

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
            '--dataset_mode',
            dest = 'ds_mode',
            type = str,
            required = True,
            help = 'which sub-category of dataset Market1501 is to be used'
            )
    parse.add_argument(
            '--store_pth',
            dest = 'store_pth',
            type = str,
            required = True,
            help = 'path that the embeddings are stored: e.g.: ./res/emb.pkl'
            )
    parse.add_argument(
            '--data_pth',
            dest = 'data_pth',
            type = str,
            required = True,
            help = 'path that the raw images are stored'
            )

    return parse.parse_args()



def embed(args):
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
    ds = Market1501(args.data_pth, mode = args.ds_mode)
    sampler = SequentialSampler(ds)
    dl = DataLoader(ds, sampler = sampler, num_workers = 4)
    diter = iter(dl)

    ## embedding samples
    logger.info('start embedding')
    all_iter_nums = len(ds) // batchsize + 1
    embeddings = []
    label_ids = []
    label_cams = []
    it = 0
    # TODO: refine this dataloader and make this faster, better get rid of is_finish
    is_finish = False
    while True:
        it += 1
        if is_finish: break
        # get one batch data
        imgs = []
        for i in range(batchsize):
            try:
                img, lb_id, lb_cam = next(diter)
                label_ids.append(lb_id)
                label_cams.append(lb_cam)
                imgs.append(img)
            except StopIteration:
                is_finish = True
                break
        imgs = torch.cat(imgs)

        print('\r=======>  processing iter {} / {}'.format(it, all_iter_nums), end = '', flush = True)
        imgs = imgs.cuda()
        _, _, C, H, W = imgs.shape
        imgs = imgs.contiguous().view(-1, C, H, W)
        embd = model(imgs).detach().cpu().numpy()
        embeddings.append(embd)
    print('  ...   completed')

    embeddings = np.vstack(embeddings)
    label_ids = np.hstack(label_ids)
    label_cams = np.hstack(label_cams)

    ## aggregate embeddings
    logger.info('aggregating embeddings')
    N, L = embeddings.shape
    embeddings = embeddings.reshape(int(N / 10), 10, L).mean(axis = 1)

    ## dump results
    logger.info('dump embeddings')
    embd_res = {'embeddings': embeddings, 'label_ids': label_ids, 'label_cams': label_cams}
    with open(args.store_pth, 'wb') as fw:
        pickle.dump(embd_res, fw)

    logger.info('everything finished')



if __name__ == '__main__':
    args = parse_args()
    embed(args)
