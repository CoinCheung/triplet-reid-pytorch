#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin = None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin = margin, p = 2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            ap_dist = torch.norm(anchor - pos, 2, dim = 1).view(-1)
            an_dist = torch.norm(anchor - neg, 2, dim = 1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss


class BatchHardTriplet(object):
    def __init__(self, *args, **kwargs):
        super(BatchHardTriplet, self).__init__()

    def __call__(self, embeds, labels):
        dist_mtrx = self.pdist(embeds, embeds).detach().cpu().numpy()
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
        num = labels.shape[0]
        lb_eqs = labels == labels.T
        pos_idxs = []
        neg_idxs = []
        for i in range(num):
            dist_min, dist_max = dist_mtx[i][0], dist_mtx[i][0]
            id_min, id_max = 0, 0
            for j in range(num):
                if label[i] == label[j]:
                    if i != j:
                        if dist_mtx[i][j] > dist_max:
                            dist_max = dist_mtx[i][j]
                            id_max = j
                else:
                    if dist_mtx[i][j] < dist_min:
                        dist_min = dist_mtx[i][j]
                        id_min = j
            pos_idxs.append(id_max)
            neg_idxs.append(id_min)
        neg_idxs = np.array(neg_idxs).reshape((-1, 1))
        pos_idxs = np.array(pos_idxs).reshape((-1, 1))
        pos = embeds[pos_idxs]
        neg = embeds[neg_idxs]
        return embeds, pos, neg



    def pdist(self, ten1, ten2):
        m, n = ten1.shape[0], ten2.shape[0]
        ten1_pow = torch.pow(ten1, 2).sum(dim = 1, keepdim = True).expand((m, n))
        ten2_pow = torch.pow(ten2, 2).sum(dim = 1, keepdim = True).expand((n, m)).t()
        dist_mtx = ten1_pow + ten2_pow
        dist_mtx = dist_mtx.addmm_(1, -2, ten1, ten2.t())
        dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
        return dist_mtx


if __name__ == '__main__':
    bh = BatchHardTriplet()
    bh('ddddd')
