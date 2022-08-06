import configargparse
import os
import random
from pathlib import Path

# safely load from string to dict
from ast import literal_eval

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import transforms as T


from model import NonLinearExtractor, ResExtractor, grad_reverse, Generator, Classifier, Discriminator, ResBase, BottleNeck, VGGBase, prototypical_classifier
from util import config_loading, model_handler, set_seed
from train import kmeans_train, fixbi_train, mix_pseudo_space_train, proto_net_train, s2t_shot_train, s2t_train, train, source_train, train_clf, new_source_train, mixup_train
from dataset import LabelTransformImageFolder, ImageList, TransformNormal, labeled_data_sampler, CustomSubset, FeatureSet, load_dloader, MixPseudoDataset, MixupDataset, CenterDataset, load_data, load_img_data, load_train_val_data, load_img_dset, load_img_dloader, new_load_img_dloader
from evaluation import evaluation, get_features

from sklearn.svm import LinearSVC
# from sklearn.cluster import kmeans_plusplus
import faiss


def arguments_parsing():
    p = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    p.add('--config', is_config_file=True, default='./config.yaml')
    p.add('--device', type=str, default='0')
    p.add('--mode', type=str, default='train')
    
    # choosing strategies, and models, and datasets
    p.add('--dataset', type=str, default='OfficeHome')

    p.add('--source', type=int, default=0)
    p.add('--target', type=int, default=1)
    p.add('--ratio', type=float, default=0.5)
    
    # transfer settings
    p.add('--transfer_loss_weight', type=float, default=1.0)
    
    # training settings
    p.add('--seed', type=int, default=12845)
    p.add('--bsize', type=int, default=32)
    p.add('--num_iters', type=int, default=3000)
    p.add('--warmup_iters', type=int, default=100)
    p.add('--alpha', type=float, default=0.5)
    p.add('--beta', type=float, default=0.8)
    p.add('--dim', type=int, default=1)
    p.add('--lambda_u', type=float, default=0.1)
    p.add('--eval_interval', type=int, default=50)
    p.add('--update_interval', type=int, default=10)
    # configurations
    p.add('--dataset_cfg', type=literal_eval)
    
    # optimizer
    p.add('--lr', type=float, default=1e-2)
    p.add('--final_lr', type=float, default=1e-4)
    p.add('--momentum', type=float, default=0.9)
    p.add('--weight_decay', type=float, default=5e-4)
    p.add('--T', type=float, default=0.05)
    
    # lr_scheduler
    p.add('--lr_gamma', type=float, default=3e-4)
    p.add('--lr_decay', type=float, default=0.75)
    
    # mdh
    p.add('--hash_table_name', type=str, default='config_hash_table.pkl')
    return p.parse_args()

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1).cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        return loss.mean()

class KLLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super(KLLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1).cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (targets * (torch.log(targets) - log_probs)).sum(dim=1)
        return loss.mean()

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

def save(file_name, **models):
    path = Path('./model') / file_name
    path.parent.mkdir(exist_ok=True, parents=True)
    for m, v in models.items():
        models[m] = v.state_dict()
    torch.save(models, path)
    
def load(file_name, **models):
    state_dict = torch.load(Path('./model') / file_name, map_location='cpu')
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


# -

@torch.no_grad()
def sinkhorn(scores, eps=0.05, niters=3):
    Q = torch.exp(scores / eps).T
    Q /= torch.sum(Q)
    
    K, B = Q.shape
    u, r, cc = torch.zeros(K).cuda(), torch.ones(K).cuda() / K, torch.ones(B).cuda() / B
    for _ in range(niters):
        u = torch.sum(Q, dim=1)
        Q *= (r / u).unsqueeze(1)
        Q *= (cc / torch.sum(Q, dim=0)).unsqueeze(0)
    return (Q / torch.sum(Q, dim=0, keepdim=True)).T


# +
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    set_seed(args.seed)

    if args.mode == 'source_train':
        bottleneck_dim = 512
        f = ResBase(backbone='resnet50', pretrained=True).cuda()
        b = BottleNeck(f.last_dim, bottleneck_dim, nonlinear=False).cuda()
        c = Classifier(bottleneck_dim, args.dataset['num_classes']).cuda()

        s_train_dset, s_train_loader = load_img_data(args, args.source, train=True)
        s_test_dset, s_test_loader = load_img_data(args, args.source, train=False)
        t_test_dset, t_test_loader = load_img_data(args, args.target, train=False)

        params = [
            {'params': f.parameters(), 'base_lr': args.lr*0.1, 'lr': args.lr*0.1},
            {'params': b.parameters(), 'base_lr': args.lr, 'lr': args.lr},
            {'params': c.parameters(), 'base_lr': args.lr, 'lr': args.lr}
        ]
        
        opt = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        lr_scheduler = LR_Scheduler(opt, args.num_iters)
        
        s_iter = iter(s_train_loader)
        criterion = CrossEntropyLabelSmooth(args.dataset['num_classes'])
        for i in range(1, args.num_iters+1):
            print('iteration: %03d/%03d, lr: %.4f' % (i, args.num_iters, lr_scheduler.get_lr()), end='\r')
            sx, sy = next(s_iter)
            sx, sy = sx.cuda().float(), sy.cuda().long()
            
            out = c(b(f(sx)))
            loss = criterion(out, sy)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
            
            if i % args.eval_interval == 0:
                acc = evaluation(t_test_loader, f, b, c)
                print('\naccuracy: %.2f%%' % (100*acc))
                f.train()
                b.train()
                c.train()
#         source_train(args, (s_train_loader, t_test_loader), (f, b, c), opt, lr_scheduler, verbose=True)
        
        save(f'{args.dataset["name"]}/s{args.source}_{args.seed}.pt', f=f, b=b, c=c)
    if args.mode == 'test':
        bottleneck_dim = 256
        f = ResBase(backbone='resnet50', pretrained=True)
        b = BottleNeck(f.last_dim, bottleneck_dim, nonlinear=False)
        
#         center_path = Path('./data') / args.dataset['name'] / 'source_only' / f's{args.source}_center.npy'
#         with center_path.open('rb') as center_file:
#             sc = torch.from_numpy(np.load(center_file)).float().cuda()
#         c = prototypical_classifier(sc).cuda()
        c = Classifier(bottleneck_dim, args.dataset['num_classes'])
        
        load(f'{args.dataset["name"]}/s{args.source}_{args.source + 2020}.pt', f=f, b=b, c=c)
        s_test_dset, s_test_loader = load_img_data(args, args.source, train=False)
        t_test_dset, t_test_loader = load_img_data(args, args.target, train=False)
        
        s_acc = evaluation(s_test_loader, f, b, c)
        t_acc = evaluation(t_test_loader, f, b, c)
        
        print('\nsource acc: %.2f%%' % (100*s_acc))
        print('target acc: %.2f%%' % (100*t_acc))
    if args.mode == 'get_features':
        bottleneck_dim = 256
        f = ResBase(backbone='resnet50', pretrained=True).cuda()
        b = BottleNeck(f.last_dim, bottleneck_dim, nonlinear=False).cuda()
        c = Classifier(bottleneck_dim, args.dataset['num_classes']).cuda()
        
        load(f'{args.dataset["name"]}/s{args.source}_{args.seed}.pt', f=f, b=b)
        s_test_dset, s_test_loader = load_img_data(args, args.source, train=False)
        t_test_dset, t_test_loader = load_img_data(args, args.target, train=False)
        
        sf = get_features(s_test_loader, f)
        tf = get_features(t_test_loader, f)

        sx, sy = sf[:, :-1], sf[:, -1]
        tx, ty = tf[:, :-1], tf[:, -1]

        output_path = Path(f'./data/{args.dataset["name"]}/source_only/s{args.source}_t{args.target}_{args.seed}.npz')
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, 'wb') as file:
            np.savez(file, s=sf, t=tf)
    if args.mode == 'e2e_shot':
        bottleneck_dim = 256
        f = ResBase(backbone='resnet50', pretrained=True).cuda()
        b = BottleNeck(f.last_dim, bottleneck_dim).cuda()
        c = Classifier(bottleneck_dim, args.dataset['num_classes']).cuda()

        params = [
            {'params': f.parameters(), 'base_lr': args.lr*0.1, 'lr': args.lr*0.1},
            {'params': b.parameters(), 'base_lr': args.lr, 'lr': args.lr},
            {'params': c.parameters(), 'base_lr': args.lr, 'lr': args.lr}
        ]

        opt = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        lr_scheduler = LR_Scheduler(opt, args.num_iters)
        
        s_train_dset, s_train_loader = load_img_data(args, args.source, train=True)
        s_test_dset, s_test_loader = load_img_data(args, args.source, train=False)
        t_train_dset, t_train_loader = load_img_data(args, args.target, train=True)
        t_test_dset, t_test_loader = load_img_data(args, args.target, train=False)

        s_iter = iter(s_train_loader)
        t_iter = iter(t_train_loader)
        
        criterion = CrossEntropyLabelSmooth(args.dataset['num_classes'])
        for i in range(1, args.num_iters+1):
            print('Iterations: %3d/%3d' % (i, args.num_iters), end='\r')
            sx, sy = next(s_iter)
            tx, _ = next(t_iter)
            sx, sy = sx.cuda().float(), sy.cuda().long()
            tx = tx.cuda().float()
            
            opt.zero_grad()
            
            s_out = c(b(f(sx)))
            s_loss = criterion(s_out, sy)
            
            s_loss.backward(retain_graph=True)
            opt.step()

            # for param in c.parameters():
            #     param.requires_grad = False
            
            # opt.zero_grad()
        
            # softmax_out = F.softmax(c(b(f(tx), reverse=True)), dim=1)
            # entropy = -softmax_out * torch.log(softmax_out + 1e-5)
            # entropy = torch.sum(entropy, dim=1)

            # ent_loss = torch.mean(entropy)
            # loss = args.lambda_u * ent_loss

            # opt.zero_grad()
            # loss.backward()
            # opt.step()

            opt.zero_grad()
            
            u_out = c(b(f(tx), reverse=True))
            
            soft_out = F.softmax(u_out, dim=1)
            u_loss = args.lambda_u * torch.mean(torch.sum(soft_out * (torch.log(soft_out + 1e-5)), dim=1))
            
            u_loss.backward()
            opt.step()

            lr_scheduler.step()

            # for param in c.parameters():
            #     param.requires_grad = True

            if i % args.eval_interval == 0:
                t_acc = evaluation(t_test_loader, f, b, c)
                print('\ntarget accuracy: %.2f%%' % (100*t_acc))
                f.train()
                b.train()
                c.train()
                
    if args.mode == 's2t':
        bottleneck_dim = 256
        f = ResBase(backbone='resnet50', pretrained=True).cuda()
        b = BottleNeck(f.last_dim, bottleneck_dim).cuda()
#         b = BottleNeck(f.last_dim, bottleneck_dim, nonlinear=False).cuda()
        c = Classifier(bottleneck_dim, args.dataset['num_classes']).cuda()
        
        load(f'{args.dataset["name"]}/s{args.source}_{args.source + 2020}.pt', f=f, b=b, c=c)
        
        for param in c.parameters():
            param.requires_grad = False
        
        params = [
            {'params': f.parameters(), 'base_lr': args.lr*0.1, 'lr': args.lr*0.1},
            {'params': b.parameters(), 'base_lr': args.lr, 'lr': args.lr}
            # {'params': c.parameters(), 'base_lr': args.lr, 'lr': args.lr}
        ]
        
        opt = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        lr_scheduler = LR_Scheduler(opt, args.num_iters)
        
        s_train_dset, s_train_loader = load_img_data(args, args.source, train=True)
        s_test_dset, s_test_loader = load_img_data(args, args.source, train=False)
        t_train_dset, t_train_loader = load_img_data(args, args.target, train=True)
        t_test_dset, t_test_loader = load_img_data(args, args.target, train=False)

        s_iter = iter(s_train_loader)
        t_iter = iter(t_train_loader)
        
        criterion = CrossEntropyLabelSmooth(args.dataset['num_classes'])
        for i in range(1, args.num_iters+1):
            print('Iterations: %3d/%3d' % (i, args.num_iters), end='\r')
            # sx, sy = next(s_iter)
            tx, _ = next(t_iter)
            # sx, sy = sx.cuda().float(), sy.cuda().long()
            tx = tx.cuda().float()
            
            # opt.zero_grad()
            
            # s_out = c(b(f(sx)))
            # s_loss = criterion(s_out, sy)
            
            # s_loss.backward(retain_graph=True)
            # opt.step()
            
            # opt.zero_grad()
            
            # t_out = c(b(f(tx), reverse=False))
            # soft_out = F.softmax(t_out, dim=1)
            # u_loss = args.lambda_u * torch.mean(torch.sum(soft_out * (torch.log(soft_out + 1e-5)), dim=1))
            # u_loss.backward()
            
            # opt.step()
            # lr_scheduler.step()

            softmax_out = F.softmax(c(b(f(tx))), dim=1)
            entropy = -softmax_out * torch.log(softmax_out + 1e-5)
            entropy = torch.sum(entropy, dim=1)

            ent_loss = torch.mean(entropy)

            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            ent_loss -= gentropy_loss
            loss = ent_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

            if i % args.eval_interval == 0:
                t_acc = evaluation(t_test_loader, f, b, c)
                print('\ntarget accuracy: %.2f%%' % (100*t_acc))
                f.train()
                b.train()
                # c.train()
#         s2t_shot_train(args, (s_test_loader, t_train_loader, t_test_loader), (f, c), opt)
        
#         sf = get_features(s_test_loader, f)
#         tf = get_features(t_test_loader, f)

#         output_path = Path(f'./data/{args.dataset["name"]}/s2t_shot/s{args.source}_t{args.target}_{args.seed}.npz')
#         output_path.parent.mkdir(exist_ok=True, parents=True)
#         with open(output_path, 'wb') as file:
#             np.savez(file, s=sf, t=tf)

    if args.mode == 'kmeans':
        bottleneck_dim = 256
        b = BottleNeck(in_features=bottleneck_dim, bottleneck_dim=bottleneck_dim, nonlinear=True).cuda()
        
        opt = torch.optim.SGD(b.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        path = Path('./data') / args.dataset['name'] / 'source_only' / f's{args.source}_t{args.target}.npz'
        
        s_set = FeatureSet(path, data_name='s')
        t_set = FeatureSet(path, data_name='t')
        sc = np.stack([s_set.x[s_set.y == i].mean(axis=0) for i in range(65)]).astype('float32')
        tf = t_set.x.astype('float32')
        factor = 65
#         centers, _ = kmeans_plusplus(tf, n_clusters=len(tf)//factor, random_state=args.seed)
#         kmeans = faiss.Kmeans(tf.shape[1], len(tf)//factor, niter=300, nredo=5, gpu=True, seed=args.seed)
        
        centers, _ = kmeans_plusplus(tf, n_clusters=factor, random_state=args.seed)
        kmeans = faiss.Kmeans(tf.shape[1], factor, niter=300, nredo=5, gpu=True, seed=args.seed)
        kmeans.train(tf, init_centroids=centers.astype('float32'))
        
        total_c = torch.from_numpy(kmeans.centroids).type(torch.cuda.FloatTensor)
        c = prototypical_classifier(total_c).cuda()
        
        t_set.y = kmeans.index.search(tf, 1)[1].flatten()
        
        s_test_loader = load_dloader(args, s_set, train=False)
        t_train_loader = load_dloader(args, t_set, train=True)
        t_test_loader = load_dloader(args, t_set, train=False)
        
        kmeans_train(args, (t_train_loader, t_test_loader), (b, c), opt, total_c)
        
        sf = get_features(s_test_loader, b)
        tf = get_features(t_test_loader, b)
        output_path = Path(f'./data/{args.dataset["name"]}/kmeans_direct_source_only/s{args.source}_t{args.target}_{factor}.npz')
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, 'wb') as f:
            np.savez(f, s=sf, t=tf)
    if args.mode == 'lc':
        bottleneck_dim = 512
        pf = ResExtractor('resnet34', bottleneck_dim).cuda()
        b = NonLinearExtractor(bottleneck_dim, bottleneck_dim).cuda()
        c = Classifier(bottleneck_dim, args.dataset['num_classes']).cuda()

        params = [
            {'params': b.parameters(), 'base_lr': args.lr, 'lr': args.lr},
            {'params': c.parameters(), 'base_lr': args.lr, 'lr': args.lr}
        ]

        opt = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        lr_scheduler = LR_Scheduler(opt, args.num_iters)

        label_correction_soft_labels = np.load(f'data/labels/label_correction_soft_labels/s{args.source}_t{args.target}_{args.dim}.npy')
        path = Path(args.dataset['path']) / args.dataset['domains'][args.source]
        s_train_dset = LabelTransformImageFolder(path, TransformNormal(train=True), label_correction_soft_labels)
        # s_train_dset = load_img_dset(args, args.source, train=train)
        s_train_loader = load_img_dloader(args, s_train_dset, train=True)
        s_test_dset, s_test_loader = load_img_data(args, args.source, train=False)
        
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
        
        s_iter = iter(s_train_loader)
        l_iter = iter(t_labeled_train_loader)
        u_iter = iter(t_unlabeled_train_loader)

        criterion = nn.CrossEntropyLoss()
        b.train()
        c.train()

        for i in range(1, args.num_iters+1):
            print('iteration: %03d/%03d, lr: %.4f' % (i, args.num_iters, lr_scheduler.get_lr()), end='\r')   
            lx, ly = next(l_iter)
            lx, ly = lx.float().cuda(), ly.long().cuda()
            
            # sx, sy = next(s_iter)
            # sx, sy = sx.float().cuda(), sy.long().cuda()

            sx, sy1, sy2 = next(s_iter)
            sx, sy1, sy2 = sx.float().cuda(), sy1.long().cuda(), sy2.float().cuda()
            # soft_sy = class_soft_labels[sy]
            ux, _ = next(u_iter)
            ux = ux.float().cuda()

            opt.zero_grad()
            
            # inputs, targets = torch.cat((sx, lx)), torch.cat((sy, ly))
            s_out = c(b(pf(sx)))
            # loss = criterion(s_out, sy)
            # loss = (1 - args.alpha) * criterion(s_out, sy)
            s_log_softmax_out = F.log_softmax(s_out, dim=1)
            l_loss = nn.CrossEntropyLoss(reduction='none')(s_out, sy1)
            # l_loss2 = criterion(s_out, sy2)

            # s_loss = (l_loss1 + l_loss2)/2

            soft_loss = -(sy2 * s_log_softmax_out).sum(axis=1)
            s_loss = ((1 - args.beta) * l_loss  + args.beta * soft_loss).mean()

            # soft_loss = -(global_soft_labels * s_log_softmax_out).sum(axis=1)
            # s_loss = ((1 - args.alpha) * s_loss  + args.alpha * soft_loss).mean()

            # addi = -(s_log_softmax_out/65).sum(dim=1)
            # s_loss = ((1 - args.alpha) * l_loss  + args.alpha * addi).mean()
            # s_loss = criterion(s_out, sy)
            # soft_out = F.softmax(l_out, dim=1)
            # h_loss = - torch.mean(torch.sum(soft_out * (torch.log(soft_out + 1e-5)), dim=1))
            # loss = (1 - args.lambda_u) * l_loss + args.lambda_u * h_loss
            
            # t_out = c(b(f(lx)))
            # t_loss = torch.nn.CrossEntropyLoss()(t_out, ly)

            # loss = (s_loss + t_loss)/2
            # loss = soft_loss.mean()
            loss = s_loss
            loss.backward()
            opt.step()
            
            lr_scheduler.step()

            if i % args.eval_interval == 0:
                t_acc = evaluation(t_unlabeled_test_loader, pf, b, c)
                print('\ntgt accuracy: %.2f%%' % (100*t_acc))
                b.train()
                c.train()

        save(f'{args.dataset["name"]}/{args.mode}/res34/s{args.source}_t{args.target}_{args.seed}/label_correction_{args.beta}_{args.num_iters}_{args.dim}.pt', pf=pf, b=b, c=c)
    
    if args.mode == '3shot':
        bottleneck_dim = 512
        f = ResBase(backbone='resnet34', pretrained=True).cuda()
        b = BottleNeck(f.last_dim, bottleneck_dim).cuda()
        c = Classifier(bottleneck_dim, args.dataset['num_classes']).cuda()

        # load(f'{args.dataset["name"]}/3shot/res34/s{args.source}_t{args.target}_{args.seed}/t.pt', f=f, b=b, c=c)

        # for param in c.parameters():
        #     param.requires_grad = False
        
        params = [
            {'params': f.parameters(), 'base_lr': args.lr*0.1, 'lr': args.lr*0.1},
            {'params': b.parameters(), 'base_lr': args.lr, 'lr': args.lr},
            {'params': c.parameters(), 'base_lr': args.lr, 'lr': args.lr}
        ]
        
        opt = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        lr_scheduler = LR_Scheduler(opt, args.num_iters)
        label_correction_soft_labels = np.load(f'data/labels/avg_distance/s{args.source}_t{args.target}_{args.T}.npy')
        path = Path(args.dataset['path']) / args.dataset['domains'][args.source]
        s_train_dset = LabelTransformImageFolder(path, TransformNormal(train=True), label_correction_soft_labels)
        # s_train_dset = load_img_dset(args, args.source, train=train)
        s_train_loader = load_img_dloader(args, s_train_dset, train=True)
        s_test_dset, s_test_loader = load_img_data(args, args.source, train=False)
        
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
        
        s_iter = iter(s_train_loader)
        l_iter = iter(t_labeled_train_loader)
        u_iter = iter(t_unlabeled_train_loader)
        # criterion = KLLabelSmooth(args.dataset['num_classes'], epsilon=args.alpha)
        # criterion = CrossEntropyLabelSmooth(args.dataset['num_classes'], epsilon=args.alpha)
        criterion = nn.CrossEntropyLoss()
        f.train()
        b.train()
        c.train()
        # class_soft_labels = np.load(f'data/labels/custom_soft_labels/s{args.source}_t{args.target}_6.npy')
        # class_soft_labels = torch.from_numpy(class_soft_labels).float().cuda()

        # global_soft_labels = np.load(f'data/labels/global_soft_labels/s{args.source}_t{args.target}.npy')
        # global_soft_labels = torch.from_numpy(global_soft_labels).float().cuda()
        for i in range(1, args.num_iters+1):
            
            lx, ly = next(l_iter)
            lx, ly = lx.float().cuda(), ly.long().cuda()
            
            # sx, sy = next(s_iter)
            # sx, sy = sx.float().cuda(), sy.long().cuda()

            sx, sy1, sy2 = next(s_iter)
            sx, sy1, sy2 = sx.float().cuda(), sy1.long().cuda(), sy2.float().cuda()
            # soft_sy = class_soft_labels[sy]
            ux, _ = next(u_iter)
            ux = ux.float().cuda()

            opt.zero_grad()
            
            # inputs, targets = torch.cat((sx, lx)), torch.cat((sy, ly))
            s_out = c(b(f(sx)))
            # loss = criterion(s_out, sy)
            # loss = (1 - args.alpha) * criterion(s_out, sy)
            s_log_softmax_out = F.log_softmax(s_out, dim=1)
            l_loss = nn.CrossEntropyLoss(reduction='none')(s_out, sy1)
            # l_loss2 = criterion(s_out, sy2)

            # s_loss = (l_loss1 + l_loss2)/2

            soft_loss = -(sy2 * s_log_softmax_out).sum(axis=1)
            s_loss = ((1 - args.beta) * l_loss  + args.beta * soft_loss).mean()
            # soft_loss = -(global_soft_labels * s_log_softmax_out).sum(axis=1)
            # s_loss = ((1 - args.alpha) * s_loss  + args.alpha * soft_loss).mean()
            print('iteration: %03d/%03d, lr: %.4f, loss: %.4f' % (i, args.num_iters, lr_scheduler.get_lr(), s_loss.item()), end='\r')   
            # addi = -(s_log_softmax_out/65).sum(dim=1)
            # s_loss = ((1 - args.alpha) * l_loss  + args.alpha * addi).mean()
            # s_loss = criterion(s_out, sy)
            # soft_out = F.softmax(l_out, dim=1)
            # h_loss = - torch.mean(torch.sum(soft_out * (torch.log(soft_out + 1e-5)), dim=1))
            # loss = (1 - args.lambda_u) * l_loss + args.lambda_u * h_loss
            
            # t_out = c(b(f(lx)))
            # t_loss = torch.nn.CrossEntropyLoss()(t_out, ly)

            # loss = (s_loss + t_loss)/2
            # loss = soft_loss.mean()
            loss = s_loss
            loss.backward()
            opt.step()

            # opt.zero_grad()
            # sf = b(f(sx))
            # s_log_softmax_out = F.log_softmax(c(sf.detach()), dim=1)
            # addi = -(s_log_softmax_out/65).sum(dim=1)
            # loss = args.alpha * addi.mean()
            # loss.backward()
            # opt.step()
            # for param in c.parameters():
            #     param.requires_grad = False
            
            # opt.zero_grad()
            
            # l_out = c(b(f(sx)))
            # l_loss = criterion(l_out, sy)
            
            # l_loss.backward()
            # opt.step()

            # for param in c.parameters():
            #     param.requires_grad = True
            # opt.zero_grad()
            
            # # inputs, targets = torch.cat((sx, lx)), torch.cat((sy, ly))
            # l_out = c(b(f(lx)))
            # l_loss = criterion(l_out, ly)
            
            # l_loss.backward()
            # opt.step()
            
            # opt.zero_grad()
            
            # u_out = c(b(f(ux), reverse=True))
            
            # soft_out = F.softmax(u_out, dim=1)
            # u_loss = args.lambda_u * torch.mean(torch.sum(soft_out * (torch.log(soft_out + 1e-5)), dim=1))
            
            # u_loss.backward()
            # opt.step()

            # u_out = c(b(f(ux)))

            # softmax_out = F.softmax(u_out, dim=1)
            # entropy = -softmax_out * torch.log(softmax_out + 1e-5)
            # entropy = torch.sum(entropy, dim=1)

            # ent_loss = torch.mean(entropy)

            # msoftmax = softmax_out.mean(dim=0)
            # gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            # ent_loss -= gentropy_loss
            
            # loss = args.lambda_u * ent_loss

            # opt.zero_grad()
            # loss.backward()
            # opt.step()
            
            lr_scheduler.step()

            if i % args.eval_interval == 0:
                # s_acc = evaluation(s_test_loader, f, b, c)
                t_acc = evaluation(t_unlabeled_test_loader, f, b, c)
                # print('\nsrc accuracy: %.2f%%' % (100*s_acc))
                print('\ntgt accuracy: %.2f%%' % (100*t_acc))
                f.train()
                b.train()
                c.train()

        save(f'{args.dataset["name"]}/3shot/res34/s{args.source}_t{args.target}_{args.seed}/label_correction_avg_distance_{args.beta}_{args.num_iters}_{args.T}.pt', f=f, b=b, c=c)
        # save(f'{args.dataset["name"]}/3shot/res34/s{args.source}_t{args.target}_{args.seed}/s.pt', f=f, b=b, c=c)

        # output_path = Path(f'./data/{args.dataset["name"]}/3shot/res34/s{args.source}_t{args.target}_{args.seed}/class_wise_label_smoothing_{args.alpha}.npz')
        # output_path = Path(f'./data/{args.dataset["name"]}/3shot/res34/s{args.source}_t{args.target}_{args.seed}/label_correction_soft_labels_{args.num_iters}.npz')

        # output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # sf = get_features(s_test_loader, f, b)
        # tlf = get_features(t_labeled_test_loader, f, b)
        # tuf = get_features(t_unlabeled_test_loader, f, b)
        # with open(output_path, 'wb') as file:
        #     np.savez(file, s=sf, tl=tlf, tu=tuf)

if __name__ == '__main__':
    args = arguments_parsing()
    
    # replace the configuration
    args.dataset = args.dataset_cfg[args.dataset]
    # mdh
    args.mdh = model_handler(
        Path(args.dataset['path']) / 'model', 
        args.hash_table_name
    )
    main(args)
