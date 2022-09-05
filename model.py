# -*- coding: utf-8 -*-
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


def get_optimizer(model, weight_decay, lr, momentum):
    params = model.get_parameters(init_lr=1.0)
    optimizer = torch.optim.SGD(params, weight_decay=weight_decay, lr=lr, momentum=momentum, nesterov=False)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)
    )
    return scheduler

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ResBase(nn.Module):
    def __init__(self, backbone='resnet50', output_dim=256, **kwargs):
        super(ResBase, self).__init__()
        self.res = models.__dict__[backbone](**kwargs)
        self.last_dim = self.res.fc.in_features
        self.res.fc = nn.Identity()
#         self.res.fc = nn.Sequential(
#             nn.Linear(self.res.fc.in_features, output_dim),
#             nn.BatchNorm1d(output_dim, affine=True)
#         )

    def forward(self, x):
        return self.res(x)

class VGGBase(nn.Module):
    def __init__(self, backbone='vgg16', output_dim=256, **kwargs):
        super(VGGBase, self).__init__()
        self.vgg = models.__dict__[backbone](**kwargs)
        self.vgg.classifier[6] = nn.Sequential(
            nn.Linear(self.vgg.classifier[6].in_features, output_dim),
            nn.BatchNorm1d(output_dim, affine=True)
        )

    def forward(self, x):
        return self.vgg(x)

# +
# class BottleNeck(nn.Module):
#     def __init__(self, in_features, bottleneck_dim, nonlinear=False):
#         super(BottleNeck, self).__init__()
#         if nonlinear:
#             self.bottleneck = nn.Sequential(
#                 nn.Linear(in_features, bottleneck_dim, bias=True),
#                 nn.BatchNorm1d(bottleneck_dim),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(in_features, bottleneck_dim, bias=True),
#                 nn.BatchNorm1d(bottleneck_dim),
#             )
# #             self.bottleneck[0].weight.data.copy_(torch.eye(in_features))
# #             self.bottleneck[3].weight.data.copy_(torch.eye(in_features))
#         else:
#             self.bottleneck = nn.Linear(in_features, bottleneck_dim)
#             self.bottleneck.apply(init_weights)
# #             self.bottleneck.weight.data.copy_(torch.eye(in_features))
#         self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
#     def forward(self, x):
#         return self.bn(self.bottleneck(x))
    
# class Classifier(nn.Module):
#     def __init__(self, bottleneck_dim, num_classes):
#         super(Classifier, self).__init__()
#         self.fc = weightNorm(nn.Linear(bottleneck_dim, num_classes), name='weight')
# #         self.fc = nn.Linear(bottleneck_dim, num_classes)
#     def forward(self, x):
#         return self.fc(x)
# -

class torch_prototypical_classifier(nn.Module):
    def __init__(self, center=None):
        super(torch_prototypical_classifier, self).__init__()
        self.center = None
        if center is not None:
            self.update_center(center)
    def update_center(self, c):
        self.center = c
        self.center.require_grad = False

    @torch.no_grad()
    def forward(self, x, T=1.0):
        dist = torch.cdist(x, self.center)
        return F.softmax(-dist*T, dim=1)

def prototypical_classifier(source, center, T=1.0):
    dist = cdist(source, center)
    return softmax(-dist * T, axis=1)

# +

class ResModel(nn.Module):
    def __init__(self, backbone='resnet34', bottleneck_dim=512, output_dim=65, pre_trained=True):
        super(ResModel, self).__init__()
        self.f = ResBase(backbone=backbone, weights=models.__dict__[f'ResNet{backbone[-2:]}_Weights'].DEFAULT if pre_trained else None)
        self.b = BottleNeck(self.f.last_dim, bottleneck_dim)
        self.c = Classifier(bottleneck_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
    def get_params(self, lr):
        return [
            {'params': self.f.parameters(), 'base_lr': lr*0.1, 'lr': lr*0.1},
            {'params': self.b.parameters(), 'base_lr': lr, 'lr': lr},
            {'params': self.c.parameters(), 'base_lr': lr, 'lr': lr}
        ]
    def get_features(self, x, reverse=False):
        return self.b(self.f(x), reverse=reverse)

    def forward(self, x, reverse=False):
        f = self.get_features(x, reverse=reverse)
        return self.c(f)

    def base_loss(self, x, y):
        out = self.forward(x)
        return self.criterion(out, y)

    def mme_loss(self, x, lamda=0.1):
        out = self.forward(x, reverse=True)
        out = F.softmax(out, dim=1)
        adent = lamda * torch.mean(torch.sum(out * (torch.log(out + 1e-10)), dim=1))
        return adent

    def lc_loss(self, x, y1, y2, alpha):
        out = self.forward(x)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = nn.CrossEntropyLoss(reduction='none')(out, y1)
        soft_loss = -(y2 * log_softmax_out).sum(axis=1)
        return ((1 - alpha) * l_loss + alpha * soft_loss).mean()
    def lckl_loss(self, x, y1, y2, alpha):
        out = self.forward(x)
        log_softmax_out = F.log_softmax(out, dim=1)
        target = y2.mul(alpha).scatter(dim=1, index=y1.reshape(-1, 1), value=1-alpha, reduce='add')
        kl_loss = (target * (torch.log(target) - log_softmax_out)).sum(axis=1)
        return kl_loss.mean()
    def mpd_loss(self, x, y, T, alpha):
        out = self.forward(x)
        y2 = F.softmax(out * T, dim=1)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = nn.CrossEntropyLoss(reduction='none')(out, y)
        soft_loss = -(y2 * log_softmax_out).sum(axis=1)
        return ((1 - alpha) * l_loss + alpha * soft_loss).mean()
    def targetRP_loss(self, x, y1, centers, T, alpha):
        sf = self.get_features(x)
        out = self.c(sf)
        dist = torch.cdist(sf.detach(), centers)
        y2 = F.softmax(-dist * T, dim=1)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = nn.CrossEntropyLoss(reduction='none')(out, y1)
        soft_loss = -(y2 * log_softmax_out).sum(axis=1)
        return ((1 - alpha) * l_loss + alpha * soft_loss).mean()



class ResExtractor(nn.Module):
    def __init__(self, backbone='resnet34', weights='ResNet34_Weights', bottleneck_dim=512):
        super(ResExtractor, self).__init__()
        self.f = ResBase(backbone=backbone, weights=weights)
        self.b = BottleNeck(self.f.last_dim, bottleneck_dim)
    def forward(self, x):
        return self.b(self.f(x)).detach()

class NonLinearExtractor(nn.Module):
    def __init__(self, in_features, bottleneck_dim):
        super(NonLinearExtractor, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(in_features, bottleneck_dim, bias=True),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, bottleneck_dim, bias=True),
            nn.BatchNorm1d(bottleneck_dim),
        )
        self.bottleneck.apply(init_weights)
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
    def forward(self, x):
        return self.bn(self.bottleneck(x))

class BottleNeck(nn.Module):
    def __init__(self, in_features, bottleneck_dim):
        super(BottleNeck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.bottleneck = nn.Linear(in_features, bottleneck_dim)
        self.bottleneck.apply(init_weights)
    def forward(self, x, reverse=False):
        x = self.bottleneck(x)
        if reverse:
            x = grad_reverse(x)
        return self.bn(x)

class Classifier(nn.Module):
    def __init__(self, bottleneck_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = weightNorm(nn.Linear(bottleneck_dim, num_classes))
        self.fc.apply(init_weights)
    def forward(self, x):
        return self.fc(x)
    
# -

class MME_Classifier(nn.Module):
    def __init__(self, bottleneck_dim, num_classes):
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_classes, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out


class Discriminator(nn.Module):
    def __init__(self, in_features, inner_dim=256, output_dim=2):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=inner_dim),
            nn.ReLU(),
            nn.Linear(in_features=inner_dim, out_features=output_dim)
        )

    def forward(self, x):
        return self.discriminator(x)

class Generator(nn.Module):
    def __init__(self, output_dim=256, **kwargs):
        super(Generator, self).__init__()
        self.res = ResBase(**kwargs)
        self.bottle = BottleNeck(self.res.output_dim, bottleneck_dim=output_dim, nonlinear=True)
    def forward(self, x):
        return self.bottle(self.res(x))

# +
# class Generator(nn.Module):
#     def __init__(self, input_dim=256, output_dim=512):
#         super(Generator, self).__init__()
#         self.f = nn.Sequential(
#             nn.Linear(input_dim, input_dim, bias=False),
#             nn.BatchNorm1d(input_dim),
#             nn.ReLU(inplace=True), # first layer
#             nn.Linear(input_dim, input_dim, bias=False),
#             nn.BatchNorm1d(input_dim),
#             nn.ReLU(inplace=True), # second layer
#             nn.Linear(input_dim, output_dim),
#             nn.BatchNorm1d(output_dim, affine=False) # output layer
#         ) 
#     def forward(self, x):
#         return self.f(x)
