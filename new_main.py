import configargparse
import os
import random
from pathlib import Path

# safely load from string to dict
from ast import literal_eval

import numpy as np
from scipy.special import softmax
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from mme_model import ResModel
# from model import ResModel, prototypical_classifier, torch_prototypical_classifier
from util import set_seed
from dataset import get_loaders, LabelTransformImageFolder, ImageList, TransformNormal, labeled_data_sampler, CustomSubset, FeatureSet, load_dloader, MixPseudoDataset, MixupDataset, CenterDataset, load_data, load_img_data, load_train_val_data, load_img_dset, load_img_dloader, new_load_img_dloader
from evaluation import evaluation, get_features, get_predictions
from mdh import ModelHandler

def arguments_parsing():
    p = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    p.add('--config', is_config_file=True, default='./new_config.yaml')
    p.add('--device', type=str, default='0')
    p.add('--method', type=str, default='base')
    # choosing strategies, and models, and datasets
    p.add('--dataset', type=str, default='OfficeHome')

    p.add('--source', type=int, default=0)
    p.add('--target', type=int, default=1)

    # training settings
    p.add('--seed', type=int, default=2020)
    p.add('--bsize', type=int, default=24)
    p.add('--num_iters', type=int, default=3000)
    p.add('--alpha', type=float, default=0.5)

    p.add('--eval_interval', type=int, default=500)
    p.add('--update_interval', type=int, default=10)
    p.add('--log_interval', type=int, default=100)
    # configurations
    p.add('--dataset_cfg', type=literal_eval)
    
    # optimizer
    p.add('--lr', type=float, default=1e-2)
    p.add('--momentum', type=float, default=0.9)
    p.add('--weight_decay', type=float, default=5e-4)
    p.add('--T', type=float, default=0.05)
    p.add('--note', type=str, default='')

    p.add('--init', type=str, default='')
    return p.parse_args()

class LR_Scheduler(object):
    def __init__(self, optimizer, num_iters, final_lr=None):
        # if final_lr: use cos scheduler, otherwise, use gamma scheduler
        self.final_lr = final_lr
        self.optimizer = optimizer
        self.iter = 0
        self.num_iters = num_iters
        self.current_lr = 0
    def step(self):
        for param_group in self.optimizer.param_groups:
            base = param_group['base_lr']
            self.current_lr = param_group['lr'] = (
                self.final_lr + 0.5 * (base - self.final_lr)*(1 + np.cos(np.pi * self.iter/self.num_iters))
                if self.final_lr
                else base * ((1 + 10 * self.iter / self.num_iters) ** (-0.75))
#                 else base * ((1 + 0.0001 * self.iter) ** (-0.75))
            )
        self.iter += 1
    def refresh(self):
        self.iter = 0
    def get_lr(self):
        return self.current_lr

def save(path, **models):
    for m, v in models.items():
        models[m] = v.state_dict()
    torch.save(models, path)
    
def load(path, **models):
    state_dict = torch.load(path, map_location='cpu')
    for m, v in models.items():
        v.load_state_dict(state_dict[m])

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    set_seed(args.seed)

    if args.method == 'targetRP' or args.method == 'initTargetRP' or args.method == 'MPD':
        model = ResModel('resnet34', output_dim=args.dataset['num_classes'])
        load(args.mdh.gh.getModelPath(args.init), model=model)
        model.cuda()
    else:
        model = ResModel('resnet34', output_dim=args.dataset['num_classes']).cuda()

    params = model.get_params(args.lr)
    opt = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LR_Scheduler(opt, args.num_iters)

    s_train_loader, s_test_loader, t_labeled_train_loader, t_labeled_test_loader, t_unlabeled_train_loader, t_unlabeled_test_loader = get_loaders(args)
    torch.cuda.empty_cache()

    s_iter = iter(s_train_loader)
    l_iter = iter(t_labeled_train_loader)
    u_iter = iter(t_unlabeled_train_loader)

    model.train()

    writer = SummaryWriter(args.mdh.getLogPath())
    writer.add_text('Hash', args.mdh.getHashStr())

    for i in range(1, args.num_iters+1):
        opt.zero_grad()

        if args.method == 'base':
            sx, sy1 = next(s_iter)
            sx, sy1 = sx.float().cuda(), sy1.long().cuda()
            s_loss = model.base_loss(sx, sy1)

        lx, ly = next(l_iter)
        lx, ly = lx.float().cuda(), ly.long().cuda()

        ux, _ = next(u_iter)
        ux = ux.float().cuda()

        t_loss = model.base_loss(lx, ly)
        loss = (s_loss + t_loss)/2

        loss.backward()
        opt.step()
        lr_scheduler.step()

        if i % args.log_interval == 0:
            writer.add_scalar('LR', lr_scheduler.get_lr(), i)
            writer.add_scalar('Loss/s_loss', s_loss.item(), i)
            writer.add_scalar('Loss/t_loss', t_loss.item(), i)

        if i % args.eval_interval == 0:
            s_acc = evaluation(s_test_loader, model)
            t_acc = evaluation(t_unlabeled_test_loader, model)
            
            writer.add_scalar('Acc/s_acc.', s_acc, i)
            writer.add_scalar('Acc/t_acc.', t_acc, i)
            model.train()
    save(args.mdh.getModelPath(), model=model)
if __name__ == '__main__':
    args = arguments_parsing()
    mdh = ModelHandler(args, keys=['dataset', 'method', 'source', 'target', 'seed', 'num_iters', 'alpha', 'T', 'init'])
    
    # replace the configuration
    args.dataset = args.dataset_cfg[args.dataset]
    args.mdh = mdh
    main(args)