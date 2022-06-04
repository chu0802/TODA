import torch
from torch import nn
from torch.nn.functional import softmax
import torch.nn.functional as F
from evaluation import evaluation, get_features
import numpy as np

# +
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)

    idx = torch.randperm(x.shape[0]).cuda()
    mixed_x = lam * x + (1 - lam) * x[idx, :]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)


# +
def kmeans_train(args, loaders, models, opt, total_c):
    b, c = models
    train_loader, test_loader = loaders
    t_iter = iter(train_loader)
    
    criterion = nn.CrossEntropyLoss()
    for i in range(1, args.num_iters+1):
        print('Iterations: %3d/%3d' % (i, args.num_iters), end='\r')
        tx, ty = next(t_iter)
        tx, ty = tx.cuda().float(), ty.cuda().long()
        
        t_out = c(b(tx))
        loss = criterion(t_out, ty)
#         t_dist = torch.linalg.norm(b(tx)-total_c[ty], dim=1)
#         loss = t_dist.mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if i % args.eval_interval == 0:
            t_acc = evaluation(test_loader, b, c)
            print('\ntarget accuracy: %.2f%%' % (100*t_acc))
            b.train()
            c.train()

def mix_pseudo_space_train(args, loaders, models, opt):
    f, c = models
    train_loader, test_loader = loaders
    loader = iter(train_loader)
    celoss = nn.CrossEntropyLoss(reduction='none')
    for i in range(1, args.num_iters+1):
        print('Iterations: %3d/%3d' % (i, args.num_iters), end='\r')
        x, y1, y2, y3, p1, p2, p3 = next(loader)
        x, y1, y2, y3, p1, p2, p3 = x.cuda().float(), y1.cuda().long(), y2.cuda().long(), y3.cuda().long(), p1.cuda().float(), p2.cuda().float(), p3.cuda().float()
        
        out = c(f(x))
        l1 = celoss(out, y1)
        l2 = celoss(out, y2)
        l3 = celoss(out, y3)
        
#         loss = (p1*l1 + p2*l2 + p3*l3).mean()
        loss = l1.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if i % args.eval_interval == 0:
            acc = evaluation(test_loader, f, c)     
            print('\ntarget accuracy: %.2f%%' % (100*acc))
            f.train()
            c.train()
def proto_net_train(args, loaders, models, opt):
    f, c = models
    f.train()
    
    s_train_loader, t_train_loader, s_test_loader, t_test_loader = loaders
    
    s_iter = iter(s_train_loader)
    celoss = nn.CrossEntropyLoss()
    
    for i in range(1, args.num_iters+1):
        print('Iterations: %3d/%3d' % (i, args.num_iters), end='\r')
        (sx, sy) = next(s_iter)
        sx, sy = sx.cuda().float(), sy.cuda().long()
        
        out = f(sx)
        loss = celoss(out, sy)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if i % args.update_interval == 0:
            new_s = torch.from_numpy(get_features(s_test_loader, f))
            new_sf, new_sy = new_s[:, :-1], new_s[:, -1]
            new_sc = torch.stack([new_sf[new_sy == i].mean(dim=0) for i in range(65)]).float().cuda()
            c.update_center(new_sc)
        
        if i % args.eval_interval == 0:
            s_acc = evaluation(s_test_loader, f, c)
            t_acc = evaluation(t_test_loader, f, c)
            
            print('\nsource accuracy: %.2f%%' % (100*s_acc))
            print('\ntarget accuracy: %.2f%%' % (100*t_acc))
            f.train()
            
def fixbi_train(args, loaders, models, opt, ratio=0.7):
    b, c = models
    s_train_loader, t_train_loader, s_test_loader, t_test_loader = loaders

    s_iter = iter(s_train_loader)
    t_iter = iter(t_train_loader)
    
    celoss = nn.CrossEntropyLoss()

    for i in range(1, args.num_iters+1):
        print('Iterations: %3d/%3d' % (i, args.num_iters), end='\r')

        (sx, sy), (tx, ty) = next(s_iter), next(t_iter)
        sx, sy, tx, ty = sx.cuda().float(), sy.cuda().long(), tx.cuda().float(), ty.cuda().long()
        
        t_out = c(b(tx))
        t_pred = torch.argmax(t_out, dim=1)
        mix_x = ratio * sx + (1 - ratio) * tx
        output = b(mix_x)
        loss = ratio*celoss(output, sy) + (1 - ratio)*celoss(output, t_pred)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if i % args.eval_interval == 0:
            print(f'\nloss: {loss.item()}')
            s_acc = evaluation(s_test_loader, b, c)
            t_acc = evaluation(t_test_loader, b, c)
            print('\nsource accuracy: %.2f%%' % (100*s_acc))
            print('\ntarget accuracy: %.2f%%' % (100*t_acc))
            b.train()
            c.train()
def s2t_train(args, loaders, models, opt, centers):
    b, c, d = models
    s_train_loader, t_train_loader, s_test_loader, t_test_loader = loaders
    sc, total_c = centers
    
    s_iter = iter(s_train_loader)
    t_iter = iter(t_train_loader)
    
    celoss = nn.CrossEntropyLoss()

    for i in range(1, args.num_iters+1):
        print('Iterations: %3d/%3d' % (i, args.num_iters), end='\r')
        (sx, sy), (tx, ty) = next(s_iter), next(t_iter)
        sx, sy, tx, ty = sx.cuda().float(), sy.cuda().long(), tx.cuda().float(), ty.cuda().long()
        
        s_out = b(sx)
        
        s_center_loss = torch.linalg.norm(s_out - total_c[1], dim=1).mean()
        
        s_pred = c(s_out)
        s_class_loss = celoss(s_pred, sy)
        
        loss = s_center_loss + 10*s_class_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if i % args.eval_interval == 0:
            new_s = torch.from_numpy(get_features(s_test_loader, b))
            new_sf, new_sy = new_s[:, :-1], new_s[:, -1]
            new_sc = torch.stack([new_sf[new_sy == i].mean(dim=0) for i in range(65)]).type(torch.cuda.FloatTensor)
            c.update_center(new_sc)
            new_total_sc = new_sf.mean(dim=0).type(torch.cuda.FloatTensor)
            print(f'center_dist: {torch.linalg.norm(new_total_sc - total_c[1])}')
            print(f'\ncenter_loss: {s_center_loss.item()}, class_loss: {s_class_loss.item()}')
            s_acc = evaluation(s_test_loader, b, c)
            t_acc = evaluation(t_test_loader, c)
            
            print('\nsource accuracy: %.2f%%' % (100*s_acc))
            print('\ntarget accuracy: %.2f%%' % (100*t_acc))
            b.train()
            c.train()
            
            

def s2t_shot_train(args, loaders, models, opt):
    b, c = models
    s_train_loader, t_train_loader, s_test_loader, t_test_loader = loaders

    s_iter = iter(s_train_loader)
    t_iter = iter(t_train_loader)
    
    celoss = nn.CrossEntropyLoss()

    for i in range(1, args.num_iters+1):
        print('Iterations: %3d/%3d' % (i, args.num_iters), end='\r')
#         (sx, sy), (tx, ty) = next(s_iter), next(t_iter)
#         sx, sy, tx, ty = sx.cuda().float(), sy.cuda().long(), tx.cuda().float(), ty.cuda().long()
#         s_out, t_out = b(sx), b(tx)
        
#         s_loss = torch.linalg.vector_norm(s_out - sc[sy], dim=1).mean()
#         t_loss = torch.linalg.vector_norm(t_out - sc[ty], dim=1).mean()
#         loss = (5/6)*t_loss + (1/6)*s_loss
        (sx, sy), (tx, ty) = next(s_iter), next(t_iter)
        sx, sy, tx, ty = sx.cuda().float(), sy.cuda().long(), tx.cuda().float(), ty.cuda().long()
        
        softmax_out = softmax(c(b(tx)), dim=1)
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

        if i % args.eval_interval == 0:
            print(f'\nloss: {loss.item()}')
            s_acc = evaluation(s_test_loader, b, c)
            t_acc = evaluation(t_test_loader, b, c)
            print('\nsource accuracy: %.2f%%' % (100*s_acc))
            print('\ntarget accuracy: %.2f%%' % (100*t_acc))
            b.train()
            c.train()

def mixup_train(args, loaders, models, opt, verbose=False):
    f, c = models
    train_loader, test_loader = loaders
    num_epoches = args.num_iters // len(train_loader)
    
    celoss = nn.CrossEntropyLoss()
    for epoch in range(num_epoches):
        for x, y in train_loader:
            print('Epoches: %3d/%3d' % (epoch, num_epoches), end='\r')
            x, y = x.cuda().float(), y.cuda().long()
            mx, ya, yb, lam = mixup_data(x, y)
            out = c(f(mx))
            m_loss = mixup_criterion(celoss, out, ya, yb, lam)
            
            opt.zero_grad()
            m_loss.backward()
            opt.step()

        if verbose and epoch == num_epoches - 1:
            acc = evaluation(test_loader, f, c)
            print('\naccuracy: %.2f%%' % (100*acc))
            f.train()
            c.train()


# -

def new_source_train(args, loaders, models, opt, verbose=False):
    f, c = models
    train_loader, test_loader = loaders
    num_epoches = args.num_iters // len(train_loader)
    
    celoss = nn.CrossEntropyLoss()
    for epoch in range(num_epoches):
        for x, y in train_loader:
            print('Epoches: %3d/%3d' % (epoch, num_epoches), end='\r')
            x, y = x.cuda().float(), y.cuda().long()
            out = c(f(x))
            loss = celoss(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose and epoch == num_epoches - 1:
            acc = evaluation(test_loader, f, c)
            print('\naccuracy: %.2f%%' % (100*acc))
            f.train()
            c.train()


def source_train(args, loaders, models, opt, lr_scheduler, verbose=False):
    f, c = models
    train_loader, test_loader = loaders
    s_iter = iter(train_loader)

    celoss = nn.CrossEntropyLoss()
    for i in range(1, args.num_iters+1):
        print('Iterations: %3d/%3d' % (i, args.num_iters), end='\r')
        x, y = next(s_iter)
        x, y = x.cuda().float(), y.cuda().long()

        out = c(f(x))
        loss = celoss(out, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        lr_scheduler.step()

        if verbose and i % args.eval_interval == 0:
            acc = evaluation(test_loader, f, c)
            print('\naccuracy: %.2f%%' % (100*acc))
            f.train()
            c.train()


def train_dis_step(data, models, opt, criterion):
    sx, tx = data
    gen, dis = models
    
    opt.zero_grad()
    t_out = dis(tx).reshape(-1)
    
    t_loss = criterion(t_out, torch.zeros_like(t_out).cuda())
    t_loss.backward()

    s_gen = gen(sx)
    s_out = dis(s_gen.detach()).reshape(-1)

    s_loss = criterion(s_out, torch.ones_like(s_out).cuda())

    s_loss.backward()

    d_loss = s_loss.item() + t_loss.item()

    opt.step()

    return d_loss

def train_gen_step(data, models, opts, criteria, transfer_weight=10):
    sx, sy = data
    gen, clf, dis = models
    g_opt, c_opt = opts
    bceloss, celoss = criteria
    
    
    
    s_f = gen(sx)
    s_out = dis(s_f).reshape(-1)
    s_loss = bceloss(s_out, torch.zeros_like(s_out).cuda())
    c_loss = celoss(clf(s_f), sy)
    
    g_loss = s_loss + transfer_weight*c_loss
    
    g_opt.zero_grad()
    c_opt.zero_grad()

    g_loss.backward()
    c_opt.step()
    g_opt.step()

    return g_loss.item()

def gradient_penalty(dis, sf, tf):
    alpha = torch.rand(sf.shape[0], 1).cuda()
    diff = tf - sf
    interpolates = sf + alpha * diff
    interpolates = torch.stack([interpolates, sf, tf]).requires_grad_()
    preds = dis(interpolates)

    preds.backward(torch.ones_like(preds))
    grad_norm = interpolates.grad.norm(2, dim=1)
    penalty = ((grad_norm - 1)**2).mean()

    return penalty


def train_wgan_critic_step(data, models, opt, gamma=10):
    sx, tx = data
    gen, dis = models
    
    sf = gen(sx).detach()
    gp = gradient_penalty(dis, sf, tx)
    
    was_dis = dis(sf).mean() - dis(tx).mean()
    critic_loss = -was_dis + gamma * gp
    
    opt.zero_grad()
    critic_loss.backward()
    opt.step()
    return critic_loss.item()


def train_wgan_clf_step(data, models, opts, criterion, transfer_weight=10):
    sx, sy, tx = data
    gen, clf, dis = models
    g_opt, c_opt = opts
    
    sf = gen(sx)
    s_pred = clf(sf)
    clf_loss = criterion(s_pred, sy)
    was_dis = dis(sf).mean() - dis(tx).mean()
    g_loss = was_dis + transfer_weight * clf_loss
    
    g_opt.zero_grad()
    c_opt.zero_grad()
    g_loss.backward()
    c_opt.step()
    g_opt.step()
    
    return g_loss.item()


def train_clf(args, dloaders, model, opt):
    s_train_loader, s_test_loader, t_train_loader, t_test_loader = dloaders
    s_iter, t_iter = iter(s_train_loader), iter(t_train_loader)
    celoss = nn.CrossEntropyLoss()
    for i in range(1, args.num_iters+1):
        print('Iterations: %3d/%3d' % (i, args.num_iters), end='\r')
        (sx, sy), (tx, _) = next(s_iter), next(t_iter)
        sx, sy, tx = sx.cuda().float(), sy.cuda().long(), tx.cuda().float()

        out = model(sx)
        loss = celoss(out, sy)

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if i % args.eval_interval == 0:
            s_acc = evaluation(s_test_loader, model)
            t_acc = evaluation(t_test_loader, model)

            print('\nsource acc: %.2f%%' % (100*s_acc))
            print('target acc: %.2f%%' % (100*t_acc))

# +
def train(args, dloaders, models, optimizers, critic_n=5):
    gen, clf, dis = models
    s_train_loader, s_test_loader, t_train_loader, t_test_loader = dloaders
    gen_opt, clf_opt, dis_opt = optimizers
    s_iter, t_iter = iter(s_train_loader), iter(t_train_loader)

    bceloss = nn.BCEWithLogitsLoss()
    celoss = nn.CrossEntropyLoss()

    for i in range(1, args.num_iters+1):
#         print('Iterations: %3d/%3d' % (i, args.num_iters), end='\r')
        (sx, sy), (tx, _) = next(s_iter), next(t_iter)
        sx, sy, tx = sx.cuda().float(), sy.cuda().long(), tx.cuda().float()
        
        d_loss_sum = []
        for _ in range(critic_n):
            loss = train_wgan_critic_step(
                        (sx, tx),
                        (gen, dis),
                        dis_opt,
                        gamma=10
                    )
            d_loss_sum.append(loss)
        d_loss = sum(d_loss_sum)/critic_n
        
        g_loss = train_wgan_clf_step(
            (sx, sy, tx),
            (gen, clf, dis),
            (gen_opt, clf_opt),
            celoss,
            transfer_weight=10
        )
#         d_loss = train_dis_step(
#             (sx, tx), 
#             (gen, dis), 
#             dis_opt, 
#             bceloss
#         )

#         g_loss = train_gen_step(
#             (sx, sy),
#             (gen, clf, dis),
#             (gen_opt, clf_opt),
#             (bceloss, celoss)
#         )

        print(f'Iterations: {i}/{args.num_iters}, Loss D: {d_loss}, Loss G: {g_loss}')

  
        if i % args.eval_interval == 0:
            print(f'Iterations: {i}/{args.num_iters}, Loss D: {d_loss}, Loss G: {g_loss}')
            
#         if i % args.eval_interval == 0:
            
            s_acc = evaluation(s_test_loader, gen, clf)
            t_acc = evaluation(t_test_loader, clf)

            print('\nsource acc: %.2f%%' % (100*s_acc))
            print('target acc: %.2f%%' % (100*t_acc))
