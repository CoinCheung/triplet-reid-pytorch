#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import logging
import sys


FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


class AdamOptimWrapper(object):
    '''
    A wrapper of Adam optimizer which allows to adjust the optimizing parameters
    according to the stategy presented in the paper
    '''
    def __init__(self, params, lr, wd, t0, t1, *args, **kwargs):
        super(AdamOptimWrapper, self).__init__(*args, **kwargs)
        self.base_lr = lr
        self.wd = wd
        self.t0 = t0
        self.t1 = t1
        self.step_count = 0
        self.optim = torch.optim.Adam(params,
                lr = self.base_lr,
                weight_decay = self.wd)
        #  self.lr_inc = 0.001 ** (1.0 / (self.t1 - self.t0))


    def step(self):
        self.step_count += 1
        self.optim.step()
        # adjust optimizer parameters
        if self.step_count == self.t0:
            betas_old = self.optim.param_groups[0]['betas']
            self.optim.param_groups[0]['betas'] = (0.5, 0.999)
            betas_new = self.optim.param_groups[0]['betas']
            logger.info('==> changing adam betas from {} to {}'.format(betas_old, betas_new))
            logger.info('==> start droping lr exponentially')
        elif self.t0 < self.step_count < self.t1:
            lr = self.base_lr * (0.001 ** (float(self.step_count + 1 - self.t0) / (self.t1 + 1 - self.t0)))
            #  self.learning_rate = self.learning_rate * self.lr_inc
            for pg in self.optim.param_groups:
                pg['lr'] = lr

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.optim.param_groups[0]['lr']

