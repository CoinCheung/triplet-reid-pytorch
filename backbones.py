#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torchvision

'''
As goes with pytorch pretrained models, inception_v3 requires the input image sizes to be (299, 299), while input image sizes for other pretrained models to be (224, 224)
'''


class DenseNormReLU(nn.Module):
    def __init__(self, in_feats, out_feats, *args, **kwargs):
        super(DenseNormReLU, self).__init__(*args, **kwargs)
        self.dense = nn.Linear(in_features = in_feats, out_features = out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.dense(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EmbedNetwork(nn.Module):
    def __init__(self, dims = 128, pretrained_base = True, *args, **kwargs):
        super(EmbedNetwork, self).__init__(*args, **kwargs)

        self.base = torchvision.models.resnet50(pretrained_base)
        #  self.base = torchvision.models.inception_v3(pretrained_base)
        self.base.fc = DenseNormReLU(in_feats = 2048, out_feats = 1024)
        self.embed = nn.Linear(in_features = 1024, out_features = dims)

    def forward(self, x):
        x = self.base(x)
        x = x.contiguous().view(-1, 1024)
        x = self.embed(x)
        return x



if __name__ == "__main__":
    embed_net = EmbedNetwork(pretrained_base = False)
    print(embed_net)
    #  in_tensor = torch.randn((15, 3, 299, 299))
    in_tensor = torch.randn((15, 3, 224, 224))
    print(in_tensor.shape)
    embd = embed_net(in_tensor)
    print(embd.shape)
