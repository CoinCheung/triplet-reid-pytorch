#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch


class AdamOptimWrapper(object):
    '''
    A wrapper of Adam optimizer which allows to adjust the optimizing parameters
    according to the stategy presented in the paper
    '''
    def __init__(self, params, lr, t0, t1, *args, **kwargs):
        super(AdamOptimWrapper, self).__init__(*args, **kwargs)
        self.lr0 = lr
        self.t0 = t0
        self.t1 = t1
        self.optim = torch.optim.Adam(params, self.lr0)
        self.step_count = 0
        self.lr_inc = 0.001 ** (1.0 / (self.t1 - self.t0))

    def step(self):
        self.step_count += 1
        if self.step_count < self.t0:
            self.optim.step()
        elif self.step_count == self.t0:
            self.optim.__dict__['param_groups'][0]['betas'] = (0.5, 0.999)
        else:
            self.lr_old = self.optim.__dict__['param_groups'][0]['lr']
            self.optim.__dict__['param_groups'][0]['lr'] = self.lr_old * self.lr_inc

    def zero_grad(self):
        self.optim.zero_grad()


