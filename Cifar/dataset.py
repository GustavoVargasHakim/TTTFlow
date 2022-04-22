from torch.utils.data import Dataset
import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms

'''+++++++++++++++++++++++++PyTorch Dataset++++++++++++++++++++++++++++++++++++++++++++'''
class ToyDataset(Dataset):
    def __init__(self, tensor, labels, ssl=None):
        super().__init__()
        self.X = tensor
        self.y = labels
        self.ssl = ssl

    def __len__(self):
        return self.y.size

    def __getitem__(self, idx):
        if self.ssl is not None:
            data = {'input':self.X[idx], 'label':self.y[idx], 'pretext':self.ssl[idx]}
        else:
            data = {'input':self.X[idx], 'label':self.y[idx]}
        return data

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_Cifar(filename):
    dict = unpickle(filename)
    X = dict[b'data']
    X = X.reshape(X.shape[0], 32, 32, 3, order='F')
    if b'fine_labels' in dict :
        y = dict[b'fine_labels']
    else :
        y = dict[b'labels']
    return X, np.array(y)

def read_Cifar_test_ttt(filename, args):
    NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    te_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(*NORM)])
    teset = torchvision.datasets.CIFAR10(root=filename,
                                         train=False, download=True, transform=te_transforms)
    workers = 1
    teloader = torch.utils.data.DataLoader(teset, batch_size=args.n,
											shuffle=False, num_workers=workers)
    return teset, teloader

def read_Cifar_train_ttt(filename, args):
    NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(*NORM)])
    trset = torchvision.datasets.CIFAR10(root=filename,
										train=True, download=True, transform=tr_transforms)
    workers = 1
    trloader = torch.utils.data.DataLoader(trset, batch_size=args.n,
											shuffle=True, num_workers=workers)
    return trset, trloader

def read_Cifar_C(filename, dataroot, args, level= 1):
    '''

    :param filename:
    :param level: is to wich batch it correspond in CIFAR 10
    :return:
    '''

    tesize = 10000
    teset_raw = np.load(filename)
    NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    te_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(*NORM)])
    teset_raw = teset_raw[(level - 1) * tesize: level * tesize]
    teset = torchvision.datasets.CIFAR10(root=dataroot,
                                         train=False, download=True, transform=te_transforms)
    teset.data = teset_raw
    workers = 1
    teloader = torch.utils.data.DataLoader(teset, batch_size=args.n,
                                           shuffle=False, num_workers=workers)
    return teset, teloader