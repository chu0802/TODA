from configparser import Interpolation
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from scipy.special import softmax
from scipy.spatial.distance import cdist
import torch.nn.utils.weight_norm as weightNorm

import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)

class ResBase(nn.Module):
    def __init__(self, backbone='resnet50', output_dim=256, **kwargs):
        super(ResBase, self).__init__()
        self.res = models.__dict__[backbone](**kwargs)
        self.last_dim = self.res.fc.in_features
        self.res.fc = nn.Identity()

    def forward(self, x):
        return self.res(x)

class ResClassifier(nn.Module):
    def __init__(self, inc=4096, hidden_dim=512, num_class=65, temp=0.05):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Linear(inc, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class, bias=False)
        self.temp = temp
    def forward(self, x, reverse=False):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out

class ResModel(nn.Module):
    def __init__(self, backbone='resnet34', hidden_dim=512, output_dim=65, temp=0.05, pre_trained=True):
        super(ResModel, self).__init__()
        self.f = ResBase(backbone=backbone, weights=models.__dict__[f'ResNet{backbone[6:]}_Weights'].DEFAULT if pre_trained else None)
        self.c = ResClassifier(self.f.last_dim, hidden_dim, output_dim, temp)
        self.criterion = nn.CrossEntropyLoss()
        weights_init(self.c)
    def forward(self, x, reverse=False):
        return self.c(self.f(x), reverse)
    def get_params(self, multi=0.1):
        params = []
        for key, value in dict(self.f.named_parameters()).items():
            if value.requires_grad:
                if 'classifier' not in key:
                    params += [{'params': [value], 'lr': multi}]
                else:
                    params += [{'params': [value], 'lr': multi * 10}]
        params += [{'params': self.c.parameters(), 'lr': multi*10}]
        return params
    def base_loss(self, x, y):
        return self.criterion(self.forward(x), y)

    def mme_loss(self, x, lamda=0.1):
        out = self.forward(x, reverse=True)
        out = F.softmax(out, dim=1)
        adent = lamda * torch.mean(torch.sum(out * (torch.log(out + 1e-10)), dim=1))
        return adent


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambd
        return output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)