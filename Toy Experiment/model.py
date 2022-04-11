import torch
import torch.nn as nn

import affine
import cdf

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

