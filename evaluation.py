from xml.etree.ElementPath import xpath_tokenizer_re
import torch
import numpy as np

def evaluation(loader, *models):
    g = np.random.default_rng(2487)
    arr = np.arange(65)
    g.shuffle(arr)
    shuffle_y = torch.from_numpy(arr).long().cuda()
    for m in models:
        m.eval()
    pred, true = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda().float(), y.cuda().long()
            shuffle_ty = shuffle_y[y]
            true.append(shuffle_ty)
            for m in models:
                x = m(x)
            pred.append(x.argmax(dim=1, keepdim=True))
            
    pred, true = torch.cat(pred).flatten(), torch.cat(true).flatten()
    acc = (pred == true).float().mean()
    return acc.item()

def get_features(loader, *models):
    g = np.random.default_rng(2487)
    arr = np.arange(65)
    g.shuffle(arr)
    shuffle_y = torch.from_numpy(arr).long().cuda()
    for m in models:
        m.eval()
    features, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda().float()
            for m in models:
                x = m(x)
            shuffle_ty = shuffle_y[y]
            features.append(x.detach().cpu().numpy())
            labels.append(shuffle_ty.detach().numpy())
    print(labels[0].shape)
    return np.c_[np.vstack(features), np.hstack(labels)]
