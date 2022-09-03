from xml.etree.ElementPath import xpath_tokenizer_re
import torch
import numpy as np
import torch.nn as nn

def evaluation(loader, model):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    acc, cnt, loss = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.float().cuda(), y.long().cuda()
            out = model(x)
            pred = out.argmax(dim=1)
            acc += (pred == y).float().sum().item()
            cnt += len(x)
            loss += criterion(out, y).item()
    return loss / cnt, 100 * acc / cnt

def get_features(loader, model):
    model.eval()
    features = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.cuda().float()
            x = model.get_features(x)
            features.append(x.detach().cpu().numpy())
    return np.vstack(features)
def get_predictions(loader, model):
    model.eval()
    pred = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.cuda().float()
            x = model(x)
            pred.append(x.detach().cpu().numpy())
    return np.vstack(pred)

# def get_features(loader, *models):
#     for m in models:
#         m.eval()
#     features, labels = [], []
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.cuda().float()
#             for m in models:
#                 x = m(x)
#             features.append(x.detach().cpu().numpy())
#             labels.append(y.detach().numpy())
#     print(labels[0].shape)
#     return np.c_[np.vstack(features), np.hstack(labels)]
