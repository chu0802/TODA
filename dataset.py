from pathlib import Path
from pickletools import float8

import torch
from torch.utils.data import DataLoader, Sampler, BatchSampler, RandomSampler

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from randaugment import RandAugmentMC
import numpy as np
import random
from PIL import Image

class TransformNormal(object):
    def __init__(self, mode='train'):
        self.transform = {
            'train': transforms.Compose(
                [transforms.Resize([256, 256]),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]),
            'test': transforms.Compose(
                [transforms.Resize([224, 224]),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]),
            'strong': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        }
        self.mode = mode
    def __call__(self, x):
        return self.transform[self.mode](x)

def get_loaders(args):
    root = Path(args.dataset['path'])
    s_name, t_name = args.dataset['domains'][args.source], args.dataset['domains'][args.target]

    s_idx_path = root / f'{s_name}_list.txt'

    s_train_set = ImageList(root, s_idx_path, transform=TransformNormal(mode='train'))
    s_train_loader = load_img_dloader(args, s_train_set, train=True)

    s_test_set = ImageList(root, s_idx_path, transform=TransformNormal(mode='test'))
    s_test_loader = load_img_dloader(args, s_test_set, bsize=args.bsize*2, train=False)

    if args.mode == 'uda':
        t_idx_path = root / f'{t_name}_list.txt'

        t_unlabeled_train_set = ImageList(root, t_idx_path, transform=TransformNormal(mode='train'))
        t_unlabeled_train_loader = load_img_dloader(args, t_unlabeled_train_set, bsize=args.bsize*2, train=True)
        
        t_unlabeled_test_set = ImageList(root, t_idx_path, transform=TransformNormal(mode='test'))
        t_unlabeled_test_loader = load_img_dloader(args, t_unlabeled_test_set, bsize=args.bsize*2, train=False)

        return s_train_loader, s_test_loader, t_unlabeled_train_loader, t_unlabeled_test_loader
    elif args.mode == 'ssda':
        t_train_idx_path = root / f'{t_name}_train_3.txt'
        t_test_idx_path = root / f'{t_name}_test_3.txt'

        t_labeled_train_set = ImageList(root, t_train_idx_path, transform=TransformNormal(mode='train'))
        t_labeled_train_loader = load_img_dloader(args, t_labeled_train_set, train=True)

        t_labeled_test_set = ImageList(root, t_train_idx_path, transform=TransformNormal(mode='test'))
        t_labeled_test_loader = load_img_dloader(args, t_labeled_test_set, train=False)

        if 'CDAC' in args.method:
            t_unlabeled_train_set = ImageList(root, t_test_idx_path, transform=TransformNormal(mode='train'), strong_transform=TransformNormal(mode='strong'))
        else:
            t_unlabeled_train_set = ImageList(root, t_test_idx_path, transform=TransformNormal(mode='train'))
        t_unlabeled_train_loader = load_img_dloader(args, t_unlabeled_train_set, bsize=args.bsize*2, train=True)
        
        t_unlabeled_test_set = ImageList(root, t_test_idx_path, transform=TransformNormal(mode='test'))
        t_unlabeled_test_loader = load_img_dloader(args, t_unlabeled_test_set, bsize=args.bsize*2, train=False)

        return s_train_loader, s_test_loader, t_labeled_train_loader, t_labeled_test_loader, t_unlabeled_train_loader, t_unlabeled_test_loader

def labeled_data_sampler(dset, shot=1, seed=1362):
    rng = np.random.default_rng(seed)
    labels = (
        np.array(dset.imgs, dtype=object)[:, 1]
        if 'imgs' in dset.__dict__
        else dset.y
    )
    size = len(np.unique(labels))
    idx = np.stack([rng.choice(np.where(labels == i)[0], shot) for i in range(size)]).flatten().astype(int)
    return idx, np.setdiff1d(np.arange(len(labels)), idx)


def pil_loader(path: str):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageList(Dataset):
    def __init__(self, root, idx_path, transform, strong_transform=None):
        if not isinstance(root, Path):
            root = Path(root)
        with open(idx_path, 'r') as f:
            paths = [p[:-1].split() for p in f.readlines()]
            
        self.imgs = [(root / p, int(l)) for p, l in paths]
        self.loader = pil_loader
        self.transform = transform
        self.strong_transform = strong_transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        path, label = self.imgs[idx]
        img = self.loader(path)
        if self.strong_transform:
            img1 = self.strong_transform(img)
            img2 = self.strong_transform(img)
            return self.transform(img), label, img1, img2
        return self.transform(img), label

class LabelCorrectionImageList(ImageList):
    def __init__(self, root, idx_path, transform, target):
        super().__init__(root, idx_path, transform)
        self.target = target
    def __getitem__(self, idx):
        path, label = self.imgs[idx]
        return self.transform(self.loader(path)), label, self.target[idx]

class CustomSubset(Dataset):
    def __init__(self, dset, idx, transform=None):
        self.imgs = dset.imgs
        self.transform = transform if transform else dset.transform
        self.loader = dset.loader
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, idx):
        path, label = self.imgs[self.idx[idx]]
        return self.transform(self.loader(path)), label


class MixupDataset(Dataset):
    def __init__(self, dataset, idxs, size):
        self.transform = dataset.transform
        self.loader = dataset.loader
        self.imgs = np.array(dataset.imgs, dtype=object)[:, 0]
        self.idxs = idxs
        self.size = size
    def __len__(self):
        return len(self.idxs)*self.size
    def __getitem__(self, idx):
        idx = idx//self.size
        paths = np.random.choice(self.imgs[self.idxs[idx]], 2)
        return torch.stack([self.transform(self.loader(path)) for path in paths]).mean(dim=0), idx


class CenterDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.transform = dataset.transform
        self.loader = dataset.loader
        self.imgs = np.array(dataset.imgs, dtype=object)
        self.idxs = idxs
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, idx):
        paths = self.imgs[self.idxs[idx]]
        return torch.stack([self.transform(self.loader(path)) for path, _ in paths]).mean(dim=0), idx


class MixPseudoDataset(Dataset):
    def __init__(self, dataset, labels, probs):
        self.transform = dataset.transform
        self.loader = dataset.loader
        self.imgs = dataset.imgs
        self.labels = labels
        self.probs = probs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        path, _ = self.imgs[idx]
        return self.transform(self.loader(path)), *self.labels[idx], *self.probs[idx]


class FeatureSet(Dataset):
    def __init__(self, path, data_name=None):
        data_name = data_name if data_name else 's'
        self.data = np.load(path)[data_name]
        self.x, self.y = self.data[:, :-1], self.data[:, -1]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class LabelTransformImageFolder(ImageFolder):
    def __init__(self, path, transform, targets):
        super().__init__(path, transform)
        self.targets = targets
    def __getitem__(self, idx):
        path, org_target = self.samples[idx]
        target = self.targets[idx]
        return self.transform(self.loader(path)), org_target, target

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def load_dloader(args, dset, train=True):
    g = get_generator(args.seed)
    if train:
        dloader = InfiniteDataLoader(dset,
            batch_size = args.bsize,
            worker_init_fn=seed_worker, generator=g,
            drop_last=True, num_workers=4)
    else:
        dloader = DataLoader(dset, 
            batch_size = args.bsize,
            worker_init_fn=seed_worker, generator=g, 
            shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    
    return dloader


def load_data(args, path, data_name=None, train=True):
    dset = FeatureSet(path, data_name=data_name)
    dloader = load_dloader(args, dset, train=train)
    return dset, dloader



def load_img_dset(args, domain, train=True):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    }
    
    path = Path(args.dataset['path']) / args.dataset['domains'][domain]
    dset = ImageFolder(path, transform=transform['train' if train else 'test'])
    return dset


def load_img_dloader(args, dset, bsize=None, train=True):
    g = get_generator(args.seed)
    if train:
        dloader = InfiniteDataLoader(dset,
            batch_size = args.bsize,
            worker_init_fn=seed_worker, generator=g,
            drop_last=True, num_workers=4)
    else:
        dloader = DataLoader(dset, 
            batch_size = bsize if bsize else args.bsize,
            worker_init_fn=seed_worker, generator=g, 
            shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    return dloader


def new_load_img_dloader(args, dset, train=True):
    g = get_generator(args.seed)
    dloader = DataLoader(dset, 
        batch_size = args.bsize,
        worker_init_fn=seed_worker, generator=g, 
        shuffle=train, drop_last=train, num_workers=4, pin_memory=True)
    return dloader


def load_img_data(args, domain, train=True):
    dset = load_img_dset(args, domain, train=train)

    dloader = load_img_dloader(args, dset, train=train)
    
    return dset, dloader

def load_train_val_data(args, domain, val_rate=0.2):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    }
    
    path = Path(args.dataset['path']) / args.dataset['domains'][domain]
    dset = ImageFolder(path, transform=transform['train'])
    total_size = len(dset)
    val_size = int(total_size * val_rate)
    train_set, valid_set = torch.utils.data.random_split(dset, [total_size - val_size, val_size])

    g = get_generator(args.seed)

    train_loader = InfiniteDataLoader(train_set,
        batch_size = args.bsize,
        worker_init_fn=seed_worker, generator=g,
        drop_last=True, num_workers=4)
    val_loader = DataLoader(valid_set, 
        batch_size = args.bsize,
        worker_init_fn=seed_worker, generator=g, 
        shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

class _InfiniteSampler(Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, 
                 worker_init_fn=None, 
                 generator=None, drop_last=True, 
                 num_workers=4):
        
        sampler = RandomSampler(dataset, replacement=False, generator=generator)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)

        self._infinite_iterator = iter(DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            worker_init_fn=worker_init_fn
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0