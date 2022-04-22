import torch
import torch.nn as nn
import sys

sys.path.append('../')

from models.ResNet import ResNetCifar as ResNet
from models.SSHead import ExtractorHead
from Flow.planar import PlanarFlow

def build_model(args, device):
    print('Building model...')
    classes = 10
    norm_layer = nn.BatchNorm2d

    depth=26
    width=1
    net = ResNet(depth, width, channels=3, classes=classes, norm_layer=norm_layer).to(device)


    shared = None

    if shared == 'layer3' or shared is None:
        from models.SSHead import extractor_from_layer3
        ext = extractor_from_layer3(net)
        #head = nn.Linear(64 * width, 4)
        head = PlanarFlow(64*width)
    elif shared == 'layer2':
        from models.SSHead import extractor_from_layer2, head_on_layer2
        ext = extractor_from_layer2(net)
        # head = head_on_layer2(net, width, 4)
        head = PlanarFlow(64 * width)
    uns = ExtractorHead(ext, head).to(device)

    if hasattr(args, 'parallel') and args.parallel:
        net = torch.nn.DataParallel(net)
        uns = torch.nn.DataParallel(uns)
    return net, ext, head, uns


class net_cdf(nn.Module):
    def __init__(self, n_components=3,  nhidden=8, freeze=False, norm=False, test=False):
        super(net_cdf,self).__init__()
        self.test=test
        self.freeze = freeze
        if norm :
            self.encoder = nn.Sequential(
                nn.Linear(2, nhidden),
                nn.BatchNorm1d(nhidden),
                nn.ReLU(inplace=True),
                nn.Linear(nhidden, 2),
                nn.BatchNorm1d(2)
            )
        else :
            self.encoder = nn.Sequential(
                nn.Linear(2, nhidden),
                nn.ReLU(inplace=True),
                nn.Linear(nhidden, 2)
            )
        self.cls = nn.Sequential(
            nn.Linear(2, nhidden),
            nn.ReLU(inplace=True),
            nn.Linear(nhidden,1)
        )
        self.flow = cdf.Flow2d(n_components)
        if self.freeze :
            self.cls.requires_grad = False
            self.flow.requires_grad = False

    def forward(self, x):
        h = self.encoder(x)
        out = self.cls(h)
        if not self.test:
            z, dz_by_dx = self.flow(h)
            return out, (z, dz_by_dx)
        return out


class net_affine(nn.Module):
    def __init__(self, b_size, nhidden=8, freeze=False, norm=False, test=False):
        super(net_affine, self).__init__()
        self.test= test
        self.freeze = freeze
        if norm:
            self.encoder = nn.Sequential(
                nn.Linear(2, nhidden),
                nn.BatchNorm1d(nhidden),
                nn.ReLU(inplace=True),
                nn.Linear(nhidden, 2),
                nn.BatchNorm1d(2))
        else:
            self.encoder = nn.Sequential(
                nn.Linear(2, nhidden),
                nn.ReLU(inplace=True),
                nn.Linear(nhidden, 2))
        self.cls = nn.Sequential(
            nn.Linear(2, nhidden),
            nn.BatchNorm1d(nhidden),
            nn.ReLU(inplace=True),
            nn.Linear(nhidden,1))
        #nn.InstanceNorm1d(2)
        self.flow = affine.Flow2d(b_size)
        if self.freeze:
            self.cls.requires_grad = False
            self.flow.requires_grad = False

    def forward(self, x):
        h = self.encoder(x)
        out = self.cls(h)
        if not self.test:
            z, dz_by_dx = self.flow(h)
            return out, (z, dz_by_dx)
        return out


class net_affine_IN(nn.Module):
    def __init__(self, b_size, nhidden=8, freeze=False, norm=False, test=False):
        super(net_affine_IN, self).__init__()
        self.test= test
        self.freeze = freeze
        if norm:
            self.encoder = nn.Sequential(
                nn.Linear(2, nhidden),
                nn.BatchNorm1d(nhidden),
                nn.ReLU(inplace=True),
                nn.Linear(nhidden, 2))
                #nn.BatchNorm1d(2))
        else:
            self.encoder = nn.Sequential(
                nn.Linear(2, nhidden),
                nn.ReLU(inplace=True),
                nn.Linear(nhidden, 2))
        self.cls = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Linear(2, nhidden),
            nn.BatchNorm1d(nhidden),
            nn.ReLU(inplace=True),
            nn.Linear(nhidden,1))
        self.instance = nn.InstanceNorm1d(2)
        self.flow = affine.Flow2d(b_size)
        if self.freeze:
            self.instance.requires_grad = False
            self.cls.requires_grad = False
            self.flow.requires_grad = False

    def forward(self, x):
        h = self.encoder(x)
        out = self.cls(h)
        if not self.test:
            b = self.instance(h)
            z, dz_by_dx = self.flow(b)
            return out, (z, dz_by_dx)
        return out

