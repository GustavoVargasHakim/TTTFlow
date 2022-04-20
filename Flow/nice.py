import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden):
        '''
        :param in_out_dim: Input/output dimensions
        :param mid_dim: Number of units in hidden layers
        :param hidden:  Number of hidden layers
        '''
        super().__init__()
        self.in_block = nn.Sequential(nn.Linear(in_out_dim // 2, mid_dim),
            nn.ReLU())
        self.mid_block = nn.ModuleList([nn.Sequential(nn.Linear(mid_dim, mid_dim),
            nn.ReLU()) for _ in range(hidden - 1)])
        self.out_block = nn.Linear(mid_dim, in_out_dim // 2)

    def forward(self, x):
        x = self.in_block(x)
        for block in self.mid_block:
            x = block(x)
        x = self.out_block(x)
        return x


class CouplingLayer(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        '''
        :param in_out_dim: Input/output dimensions
        :param mid_dim: Number of units in hidden layers
        :param hidden:  Number of hidden layers
        :param mask_config: 1 if transforming odd units, 0 if transforming even units
        '''
        super().__init__()
        self.mask_config = mask_config
        self.MLP = MLP(in_out_dim, mid_dim, hidden)

    def forward(self, z):
        [B, W] = z.size()
        odd_index = torch.tensor([i for i in range(z.shape[1]) if i % 2 != 0])
        even_index = torch.tensor([i for i in range(z.shape[1]) if i % 2 == 0])
        if self.mask_config == 1:
            z1, z2 = torch.index_select(z, dim=1, index=odd_index), torch.index_select(z, dim=1, index=even_index)
        else:
            z2, z1 = torch.index_select(z, dim=1, index=odd_index), torch.index_select(z, dim=1, index=even_index)
        shift = self.MLP(z2)
        z2 = z2 + shift
        if self.mask_config == 1:
            z = torch.stack((z1, z2), dim=2).reshape((B, W))
        else:
            z = torch.stack((z2, z1), dim=2).reshape((B, W))

        return z


class Scaling(nn.Module):
    def __init__(self, dim):
        '''
        :param dim: Input/output dimensions
        '''
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(torch.zeros((1, dim)), requires_grad=True)

    def forward(self, x):
        log_det_J = torch.sum(self.scale)
        x = x * torch.exp(-self.scale)
        return x, log_det_J


class NiceFlow(nn.Module):
    def __init__(self, coupling, in_out_dim, mid_dim, hidden, mask_config):
        '''
        :param coupling: Number of coupling layers
        :param in_out_dim: Input/output dimensions
        :param mid_dim: Number of units in hidden layers
        :param hidden: Number of hidden layers
        :param mask_config: 1 if transforming odd units, 0 if transforming even units
        '''
        super(NiceFlow, self).__init__()
        self.in_out_dim = in_out_dim
        self.coupling = nn.ModuleList([
            CouplingLayer(in_out_dim=in_out_dim,
                mid_dim=mid_dim,
                hidden=hidden,
                mask_config=(mask_config + i) % 2) for i in range(coupling)])
        self.scaling = Scaling(in_out_dim)

    def forward(self, z):
        for flow in self.coupling:
            z = flow(z)
        z, log_det_J = self.scaling(z)
        return z, log_det_J


'''
print(x.shape)
flow = NiceFlow(2,48,3,3,1)
z, log_det = flow(x)
print(z.shape)'''
