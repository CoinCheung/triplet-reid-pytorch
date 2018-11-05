#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.model_zoo as model_zoo


'''
As goes with pytorch pretrained models, inception_v3 requires the input image sizes to be (299, 299), while input image sizes for other pretrained models to be (224, 224)
'''

param_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'


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
        self.pretrained_base = pretrained_base

        resnet50 = torchvision.models.resnet50()
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        self.fc_head = DenseNormReLU(in_feats = 2048, out_feats = 1024)
        self.embed = nn.Linear(in_features = 1024, out_features = dims)

        if self.pretrained_base:
            new_state = model_zoo.load_url(param_url)
            state_dict = self.state_dict()
            for k, v in new_state.items():
                if 'fc' in k: continue
                state_dict.update({k: v})
            self.load_state_dict(state_dict)

        for el in self.fc_head.children():
            if isinstance(el, nn.Linear):
                nn.init.kaiming_normal_(el.weight, a=1)
                nn.init.constant_(el.bias, 0)

        nn.init.kaiming_normal_(self.embed.weight, a=1)
        nn.init.constant_(self.embed.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.contiguous().view(-1, 2048)
        x = self.fc_head(x)
        x = self.embed(x)
        return x



if __name__ == "__main__":
    embed_net = EmbedNetwork(pretrained_base = True)

    in_ten = torch.randn(32, 3, 256, 128)
    out = embed_net(in_ten)
    print(out.shape)


