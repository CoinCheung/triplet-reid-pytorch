#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os
import logging
import time


def get_logger():
    if not os.path.exists('./res'): os.makedirs('./res')
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    logfile = 'triplet_reid-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = os.path.join('res', logfile)
    logging.basicConfig(level=logging.INFO, format=FORMAT, filename=logfile)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    return logger

logger = get_logger()
