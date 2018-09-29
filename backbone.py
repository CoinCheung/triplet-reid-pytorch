#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
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

        resnet50 = torchvision.models.resnet50(pretrained_base)
        self.base = nn.Sequential(resnet50.conv1,
                                resnet50.bn1,
                                resnet50.relu,
                                resnet50.maxpool,
                                resnet50.layer1,
                                resnet50.layer2,
                                resnet50.layer3,
                                resnet50.layer4,)

        #  self.base = torchvision.models.inception_v3(pretrained_base)
        self.fc_head = DenseNormReLU(in_feats = 2048, out_feats = 1024)
        self.embed = nn.Linear(in_features = 1024, out_features = dims)

    def forward(self, x):
        x = self.base(x)
        _, _, h, w = x.shape
        x = F.avg_pool2d(x, (h, w))
        x = x.contiguous().view(-1, 2048)
        x = self.fc_head(x)
        x = self.embed(x)
        return x



if __name__ == "__main__":
    embed_net = EmbedNetwork(pretrained_base = True)
    print(embed_net)
    #  #  in_tensor = torch.randn((15, 3, 299, 299))
    #  in_tensor = torch.randn((15, 3, 224, 224))
    #  #  print(in_tensor.shape)
    #  embd = embed_net(in_tensor)
    #  print(embd.shape)
    #  print(embed_net.state_dict().keys())
    #  #  print(embed_net.base[0].weight)
    #  net = torchvision.models.resnet50(False)
    #  #  print(net.conv1.weight)
    #  print(torch.sum(embed_net.base[0].weight == net.conv1.weight))
    #  print(embed_net.base[0].weight.shape)
    #  print(net.conv1.weight.shape)

    for i, ly in enumerate(embed_net.base):
        print(ly.__class__.__name__)
        if i > 4: break
        #  break


