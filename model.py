# -*- coding: utf-8 -*-
from configparser import Interpolation
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import torch.nn.utils.weight_norm as weightNorm

import numpy as np

# +
class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_=1.0):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input, None

grad_reverse = RevGrad.apply


# -

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

class prototypical_classifier(nn.Module):
    def __init__(self, center):
        super(prototypical_classifier, self).__init__()
        self.center = None
        self.update_center(center)
    def update_center(self, c):
        self.center = c
        self.center.require_grad = False
    def forward(self, x):
        dist = torch.cdist(x, self.center)
        return -dist

# +

class ResModel(nn.Module):
    def __init__(self, backbone='resnet34', bottleneck_dim=512, output_dim=65):
        super(ResModel, self).__init__()
        self.f = ResBase(backbone=backbone, pretrained=True)
        self.b = BottleNeck(self.f.last_dim, bottleneck_dim)
        self.c = Classifier(bottleneck_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
    def get_params(self, lr):
        return [
            {'params': self.f.parameters(), 'base_lr': lr*0.1, 'lr': lr*0.1},
            {'params': self.b.parameters(), 'base_lr': lr, 'lr': lr},
            {'params': self.c.parameters(), 'base_lr': lr, 'lr': lr}
        ]
    def get_features(self, x):
        return self.b(self.f(x))

    def forward(self, x):
        return self.c(self.b(self.f(x)))

    def base_loss(self, x, y):
        out = self.forward(x)
        return self.criterion(out, y)

    def lc_loss(self, x, y1, y2, alpha):
        out = self.forward(x)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = nn.CrossEntropyLoss(reduction='none')(out, y1)
        soft_loss = -(y2 * log_softmax_out).sum(axis=1)
        return ((1 - alpha) * l_loss + alpha * soft_loss).mean()

class ResExtractor(nn.Module):
    def __init__(self, backbone='resnet34', bottleneck_dim=512):
        super(ResExtractor, self).__init__()
        self.f = ResBase(backbone=backbone, pretrained=True)
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
        # return F.normalize(x)
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
