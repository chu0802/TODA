import configargparse
import os
import random
from pathlib import Path

# safely load from string to dict
from ast import literal_eval

import numpy as np
from scipy.special import softmax
from itertools import islice
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import ResModel, torch_prototypical_classifier
from util import set_seed
from dataset import get_loaders, LabelCorrectionImageList, LabelTransformImageFolder, ImageList, TransformNormal, labeled_data_sampler, CustomSubset, FeatureSet, load_dloader, MixPseudoDataset, MixupDataset, CenterDataset, load_data, load_img_data, load_train_val_data, load_img_dset, load_img_dloader, new_load_img_dloader
from evaluation import evaluation, get_features, get_prediction
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
    p.add('--num_iters', type=int, default=5000)
    p.add('--alpha', type=float, default=0.8)
    p.add('--beta', type=float, default=0.5)
    p.add('--lamda', type=float, default=0.1)

    p.add('--eval_interval', type=int, default=500)
    p.add('--log_interval', type=int, default=100)
    p.add('--update_interval', type=int, default=100)
    # configurations
    p.add('--dataset_cfg', type=literal_eval)
    
    # optimizer
    p.add('--lr', type=float, default=0.01)
    p.add('--momentum', type=float, default=0.9)
    p.add('--weight_decay', type=float, default=5e-4)
    p.add('--T', type=float, default=0.4)
    p.add('--temp', type=float, default=0.05)
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
        self.current_lr = optimizer.param_groups[-1]['lr']
    def step(self):
        for param_group in self.optimizer.param_groups:
            base = param_group['base_lr']
            self.current_lr = param_group['lr'] = (
                self.final_lr + 0.5 * (base - self.final_lr)*(1 + np.cos(np.pi * self.iter/self.num_iters))
                if self.final_lr
                else base * ((1 + 10 * self.iter / self.num_iters) ** (-0.75))
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

def getPPC(args, model, label, s_loader, t_loader):
    _, t_feat = get_prediction(t_loader, model)
    _, s_feat = get_prediction(s_loader, model)

    centers = torch.vstack([t_feat[label == i].mean(dim=0) for i in range(args.dataset['num_classes'])])

    ppc = torch_prototypical_classifier(centers)

    return ppc

    # soft_labels = ppc(s_feat, args.T).detach().cpu().numpy()

    # root, s_name = Path(args.dataset['path']), args.dataset['domains'][args.source]
    # s_train_set = LabelCorrectionImageList(root, root / f'{s_name}_list.txt', TransformNormal(train=True), soft_labels)
    # s_train_loader = load_img_dloader(args, s_train_set, train=True)

    # return s_train_loader

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    set_seed(args.seed)

    model = ResModel('resnet34', output_dim=args.dataset['num_classes']).cuda()

    params = model.get_params(args.lr)
    opt = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LR_Scheduler(opt, args.num_iters)

    if 'LC' in args.method:
        s_train_loader, s_test_loader, t_labeled_train_loader, t_labeled_test_loader, t_unlabeled_train_loader, t_unlabeled_test_loader = get_loaders(args)

        model_path = args.mdh.gh.getModelPath(args.init)
        init_model = ResModel('resnet34', output_dim=args.dataset['num_classes'])
        load(model_path, model=init_model)
        init_model.cuda()

        LABEL, _ = get_prediction(t_unlabeled_test_loader, init_model)
        LABEL = LABEL.argmax(dim=1)

        ppc = getPPC(args, model, LABEL, s_test_loader, t_unlabeled_test_loader)
        
        # s_train_loader = getPPCLoader(args, model, s_test_loader, t_unlabeled_test_loader)
    else:
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

        if 'LC' in args.method:
            sx, sy = next(s_iter)
            sx, sy = sx.float().cuda(), sy.long().cuda()

            sf = model.get_features(sx)
            sy2 = ppc(sf.detach(), args.T)
            s_loss = model.lc_loss(sf, sy, sy2, args.alpha)
        else:
            sx, sy = next(s_iter)
            sx, sy = sx.float().cuda(), sy.long().cuda()
            s_loss = model.base_loss(sx, sy)

        tx, ty = next(l_iter)
        tx, ty = tx.float().cuda(), ty.long().cuda()

        ux, _ = next(u_iter)
        ux = ux.float().cuda()

        t_loss = model.base_loss(tx, ty)

        loss = args.beta * s_loss + (1-args.beta) * t_loss
        
        loss.backward()
        opt.step()

        if 'MME' in args.method:
            opt.zero_grad()
            u_loss = model.mme_loss(ux, args.lamda)
            u_loss.backward()
            opt.step()

        lr_scheduler.step()

        if i % args.log_interval == 0:
            writer.add_scalar('LR', lr_scheduler.get_lr(), i)
            writer.add_scalar('Loss/s_loss', s_loss.item(), i)
            writer.add_scalar('Loss/t_loss', t_loss.item(), i)
            if 'MME' in args.method:
                writer.add_scalar('Loss/u_loss', -u_loss.item(), i)

        if i % args.eval_interval == 0:
            # s_acc = evaluation(s_test_loader, model)
            t_acc = evaluation(t_unlabeled_test_loader, model)
            # writer.add_scalar('Acc/s_acc.', s_acc, i)
            writer.add_scalar('Acc/t_acc.', t_acc, i)
            model.train()
        if i % args.update_interval == 0 and 'LCD' in args.method:
            ppc = getPPC(args, model, LABEL, s_test_loader, t_unlabeled_test_loader)
            # s_train_loader = getPPCLoader(args, model, s_test_loader, t_unlabeled_test_loader)
            # s_iter = iter(s_train_loader)
            # next(islice(s_iter, i, None))
            model.train()

    save(args.mdh.getModelPath(), model=model)
if __name__ == '__main__':
    args = arguments_parsing()
    mdh = ModelHandler(args, keys=['dataset', 'method', 'source', 'target', 'seed', 'num_iters', 'alpha', 'T', 'init', 'note', 'update_interval'])
    
    # replace the configuration
    args.dataset = args.dataset_cfg[args.dataset]
    args.mdh = mdh
    main(args)