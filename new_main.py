import configargparse
import os
import random
from pathlib import Path

# safely load from string to dict
from ast import literal_eval

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from model import ResModel
from util import set_seed
from dataset import LabelTransformImageFolder, ImageList, TransformNormal, labeled_data_sampler, CustomSubset, FeatureSet, load_dloader, MixPseudoDataset, MixupDataset, CenterDataset, load_data, load_img_data, load_train_val_data, load_img_dset, load_img_dloader, new_load_img_dloader
from evaluation import evaluation, get_features
from mdh import ModelHandler

def arguments_parsing():
    p = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    p.add('--config', is_config_file=True, default='./new_config.yaml')
    p.add('--device', type=str, default='0')
    p.add('--mode', type=str, default='train')
    # choosing strategies, and models, and datasets
    p.add('--dataset', type=str, default='OfficeHome')

    p.add('--source', type=int, default=0)
    p.add('--target', type=int, default=1)

    # training settings
    p.add('--seed', type=int, default=12845)
    p.add('--bsize', type=int, default=32)
    p.add('--num_iters', type=int, default=3000)
    p.add('--alpha', type=float, default=0.5)

    p.add('--eval_interval', type=int, default=50)
    p.add('--update_interval', type=int, default=10)
    # configurations
    p.add('--dataset_cfg', type=literal_eval)
    
    # optimizer
    p.add('--lr', type=float, default=1e-2)
    p.add('--T', type=float, default=0.05)
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

def get_dloaders(args, data_dir='/home/chu980802/tbda/data', sseed=None, tseed = None):
    if sseed is None:
        sseed = args.seed
    if tseed is None:
        tseed = args.seed
    src_path = Path(data_dir) / args.dataset['name'] / ('s%d_%d.npz' % (args.source, sseed))
    tgt_path = Path(data_dir) / args.dataset['name'] / ('s%d_%d.npz' % (args.target, tseed))
    src_train_dset, src_train_dloader = load_data(args, src_path, train=True)
    src_test_dset, src_test_dloader = load_data(args, src_path, train=False)
    tgt_train_dset, tgt_train_dloader = load_data(args, tgt_path, train=True)
    tgt_test_dset, tgt_test_dloader = load_data(args, tgt_path, train=False)
    
    return src_train_dloader, src_test_dloader, tgt_train_dloader, tgt_test_dloader

def get_img_dloaders(args):
    src_train_dset, src_train_dloader = load_img_data(args, args.source, train=True)
    src_test_dset, src_test_dloader = load_img_data(args, args.source, train=False)
    tgt_train_dset, tgt_train_dloader = load_img_data(args, args.target, train=True)
    tgt_test_dset, tgt_test_dloader = load_img_data(args, args.target, train=False)
    
    return src_train_dloader, src_test_dloader, tgt_train_dloader, tgt_test_dloader

def save(path, **models):
    for m, v in models.items():
        models[m] = v.state_dict()
    torch.save(models, path)
    
def load(path, **models):
    state_dict = torch.load(path, map_location='cpu')
    for m, v in models.items():
        v.load_state_dict(state_dict[m])
        v.cuda()
        
def normalize(x):
    return (x - x.mean(axis=0))/x.std(axis=0)

def load_features(path):
    data = np.load(path)
    ss, st = data['s'], data['t']
    ssx, ssy = normalize(ss[:, :-1]), ss[:, -1]
    stx, sty = normalize(st[:, :-1]), st[:, -1]
    return ssx, ssy, stx, sty

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    set_seed(args.seed)

    bottleneck_dim = 512
    model = ResModel('resnet34', bottleneck_dim, args.dataset['num_classes']).cuda()
    params = model.get_params(args.lr)
    opt = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LR_Scheduler(opt, args.num_iters)

    label_correction_soft_labels = np.load(f'data/labels/avg_distance/ideal/s{args.source}_t{args.target}_{args.T}.npy')
    path = Path(args.dataset['path']) / args.dataset['domains'][args.source]
    s_train_dset = LabelTransformImageFolder(path, TransformNormal(train=True), label_correction_soft_labels)
    s_train_loader = load_img_dloader(args, s_train_dset, train=True)
    s_test_dset, s_test_loader = load_img_data(args, args.source, train=False)

    
    if args.mode == 'uda':
        t_unlabeled_train_dset, t_unlabeled_train_loader = load_img_data(args, args.target, train=True)
        t_unlabeled_test_dset, t_unlabeled_test_loader = load_img_data(args, args.target, train=False)
    elif args.mode == 'ssda':
        root, t_name = Path(args.dataset['path']), args.dataset['domains'][args.target]
        t_train_idx_path = root / f'{t_name}_train_3.txt'
        t_test_idx_path = root / f'{t_name}_test_3.txt'

        t_labeled_train_set = ImageList(root, t_train_idx_path, transform=TransformNormal(train=True))
        t_labeled_train_loader = load_img_dloader(args, t_labeled_train_set, train=True)

        t_labeled_test_set = ImageList(root, t_train_idx_path, transform=TransformNormal(train=False))
        t_labeled_test_loader = load_img_dloader(args, t_labeled_test_set, train=False)

        t_unlabeled_train_set = ImageList(root, t_test_idx_path, transform=TransformNormal(train=True))
        t_unlabeled_train_loader = load_img_dloader(args, t_unlabeled_train_set, bsize=args.bsize, train=True)
        
        t_unlabeled_test_set = ImageList(root, t_test_idx_path, transform=TransformNormal(train=False))
        t_unlabeled_test_loader = load_img_dloader(args, t_unlabeled_test_set, train=False)

        l_iter = iter(t_labeled_train_loader)
    s_iter = iter(s_train_loader)
    u_iter = iter(t_unlabeled_train_loader)

    model.train()

    writer = SummaryWriter(args.mdh.getLogPath())

    for i in range(1, args.num_iters+1):
        sx, sy1, sy2 = next(s_iter)
        sx, sy1, sy2 = sx.float().cuda(), sy1.long().cuda(), sy2.float().cuda()
        
        ux, _ = next(u_iter)
        ux = ux.float().cuda()

        opt.zero_grad()

        if args.mode == 'uda':
            loss = model.lc_loss(sx, sy1, sy2, args.alpha)
            info = 's_loss: %.4f' % (loss.item())
        elif args.mode == 'ssda':
            lx, ly = next(l_iter)
            lx, ly = lx.float().cuda(), ly.long().cuda()
            s_loss = model.lc_loss(sx, sy1, sy2, args.alpha)
            t_loss = model.base_loss(lx, ly)
            loss = (s_loss + t_loss)/2
            info = 's_loss: %.4f, t_loss %.4f' % (s_loss.item(), t_loss.item())
        loss.backward()
        opt.step()
        lr_scheduler.step()
        print('iteration: %03d/%03d, lr: %.4f, %s' % (i, args.num_iters, lr_scheduler.get_lr(), info), end='\r')
        if i % args.eval_interval == 0:
            s_acc = evaluation(s_test_loader, model)
            t_acc = evaluation(t_unlabeled_test_loader, model)
            print('\nsrc accuracy: %.2f%%' % (100*s_acc))
            print('\ntgt accuracy: %.2f%%' % (100*t_acc))
            # writer.add_scalar('Loss/src', s_loss.item(), i)
            # writer.add_scalar('Loss/tgt', t_loss.item(), i)
            # writer.add_scalar('Info/LR', lr_scheduler.get_lr(), i)
            writer.add_scalar('Acc/s_acc.', s_acc, i)
            writer.add_scalar('Acc/t_acc.', t_acc, i)
            model.train()
    save(args.mdh.getModelPath(), model=model)
if __name__ == '__main__':
    args = arguments_parsing()
    mdh = ModelHandler(args, keys=['dataset', 'mode', 'source', 'target', 'seed', 'num_iters', 'alpha', 'T'])
    
    # replace the configuration
    args.dataset = args.dataset_cfg[args.dataset]
    args.mdh = mdh
    main(args)