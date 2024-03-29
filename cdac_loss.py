import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def cdac_loss(args, model, ux, ux1, ux2, target, BCE, w_cons):
    """ Get losses for unlabeled samples."""
    f = model.f(ux)
    f1 = model.f(ux1)
    f2 = model.f(ux2)

    out = model.c(f, reverse=True)
    out1 = model.c(f1, reverse=True)
    prob, prob1 = F.softmax(out, dim=1), F.softmax(out1, dim=1)

    aac_loss = advbce_unlabeled(args, target=target, f=f, prob=prob, prob1=prob1, bce=BCE)
    
    out = model.c(f)
    out1 = model.c(f1)
    out2 = model.c(f2)

    prob, prob1, prob2 = [F.softmax(i, dim=1) for i in [out, out1, out2]]
    mp, pl = torch.max(prob.detach(), dim=1)
    mask = mp.ge(0.95).float()

    pl_loss = (F.cross_entropy(out2, pl, reduction='none') * mask).mean()

    con_loss = w_cons * F.mse_loss(prob1, prob2)

    return aac_loss, pl_loss, con_loss

def advbce_unlabeled(args, target, f, prob, prob1, bce):
    """ Construct adversarial adpative clustering loss."""
    target_ulb = pairwise_target(args, f, target)
    prob_bottleneck_row, _ = PairEnum2D(prob)
    _, prob_bottleneck_col = PairEnum2D(prob1)
    adv_bce_loss = -bce(prob_bottleneck_row, prob_bottleneck_col, target_ulb)
    return adv_bce_loss

def pairwise_target(args, f, target):
    """ Produce pairwise similarity label."""
    fd = f.detach()
    # For unlabeled data
    if target is None:
        rank_feat = fd
        rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
        rank_idx1, rank_idx2 = PairEnum2D(rank_idx)
        rank_idx1, rank_idx2 = rank_idx1[:, :5], rank_idx2[:, :5]
        rank_idx1, _ = torch.sort(rank_idx1, dim=1)
        rank_idx2, _ = torch.sort(rank_idx2, dim=1)
        rank_diff = rank_idx1 - rank_idx2
        rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
        target_ulb = torch.ones_like(rank_diff).float().cuda()
        target_ulb[rank_diff > 0] = 0
    # For labeled data
    elif target is not None:
        target_row, target_col = PairEnum1D(target)
        target_ulb = torch.zeros(target.size(0) * target.size(0)).float().to(device)
        target_ulb[target_row == target_col] = 1
    else:
        raise ValueError('Please check your target.')
    return target_ulb

def PairEnum1D(x):
    """ Enumerate all pairs of feature in x with 1 dimension."""
    assert x.ndimension() == 1, 'Input dimension must be 1'
    x1 = x.repeat(x.size(0), )
    x2 = x.repeat(x.size(0)).view(-1,x.size(0)).transpose(1, 0).reshape(-1)
    return x1, x2

def PairEnum2D(x):
    """ Enumerate all pairs of feature in x with 2 dimensions."""
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    return x1, x2

def sigmoid_rampup(current, rampup_length):
    """ Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

class BCE(nn.Module):
    eps = 1e-7
    def forward(self, prob1, prob2, simi):
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()

class BCE_softlabels(nn.Module):
    """ Construct binary cross-entropy loss."""
    eps = 1e-7
    def forward(self, prob1, prob2, simi):
        P = prob1.mul_(prob2)
        P = P.sum(1)
        neglogP = - (simi * torch.log(P + BCE.eps) + (1. - simi) * torch.log(1. - P + BCE.eps))
        return neglogP.mean()
