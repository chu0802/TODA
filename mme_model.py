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
        x = self.get_features(x, reverse=reverse)
        return self.get_predictions(x)
    def get_features(self, x, reverse=False):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x)
        return F.normalize(x) / self.temp
    def get_predictions(self, x):
        return self.fc2(x)

class ResModel(nn.Module):
    def __init__(self, backbone='resnet34', hidden_dim=512, output_dim=65, temp=0.05, pre_trained=True):
        super(ResModel, self).__init__()
        self.f = ResBase(backbone=backbone, weights=models.__dict__[f'ResNet{backbone[6:]}_Weights'].DEFAULT if pre_trained else None)
        self.c = ResClassifier(self.f.last_dim, hidden_dim, output_dim, temp)
        self.criterion = nn.CrossEntropyLoss()
        weights_init(self.c)
    def forward(self, x, reverse=False):
        return self.c(self.f(x), reverse)
    def get_params(self, lr):
        params = []
        for k, v in dict(self.f.named_parameters()).items():
            if v.requires_grad:
                if 'classifier' not in k:
                    params += [{'params': [v], 'base_lr': lr*0.1, 'lr': lr*0.1}]
                else:
                    params += [{'params': [v], 'base_lr': lr, 'lr': lr}]
        params += [{'params': self.c.parameters(), 'base_lr': lr, 'lr': lr}]
        return params
    def get_features(self, x, reverse=False):
        return self.c.get_features(self.f(x), reverse=reverse)
    
    def get_predictions(self, x):
        return self.c.get_predictions(x)

    def base_loss(self, x, y):
        return self.criterion(self.forward(x), y)
    
    def lc_loss(self, f, y1, y2, alpha):
        out = self.get_predictions(f)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = nn.CrossEntropyLoss(reduction='none')(out, y1)
        soft_loss = -(y2 * log_softmax_out).sum(axis=1)
        return l_loss, soft_loss
        # return ((1 - alpha) * l_loss + alpha * soft_loss).mean()


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