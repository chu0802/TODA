from xml.etree.ElementPath import xpath_tokenizer_re
import torch
import numpy as np

def evaluation(loader, *models):
    for m in models:
        m.eval()
    pred, true = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda().float(), y.cuda().long()
            true.append(y)
            for m in models:
                x = m(x)
            pred.append(x.argmax(dim=1, keepdim=True))
            
    pred, true = torch.cat(pred).flatten(), torch.cat(true).flatten()
    acc = (pred == true).float().mean()
    return acc.item()

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
