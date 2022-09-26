import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import numpy as np

class Preprocess(nn.Module):
    def __init__(self):
        super(Preprocess, self).__init__()

    def forward(self, x, reverse=False):
        device = x.device
        if reverse:
            return (x.sigmoid() - 0.05) / 0.9
        x += Uniform(0.0, 1.0).sample(x.size()).to(device)
        x = 0.05 + 0.9 * (x / 4.0)
        z = torch.log(x) - torch.log(1-x)
        log_det_jacobian = -x.log() - (1-x).log() + torch.tensor(0.9/4).log().to(device)
        return z, log_det_jacobian

class WeightNormConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super(WeightNormConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channel,
                                                   out_channel,
                                                   kernel_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   bias=bias))

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, bottleneck):
        '''
        :param dim: input dimension (channels)
        :param bottleneck: Whether to use a bottleneck block or not
        '''
        super(ResidualBlock, self).__init__()
        self.in_block = nn.Sequential(nn.BatchNorm2d(dim),
                                      nn.ReLU())
        if bottleneck:
            self.block = nn.Sequential(WeightNormConv2d(dim, dim, 1),
                                       nn.BatchNorm2d(dim),
                                       nn.ReLU(),
                                       WeightNormConv2d(dim, dim, 3, padding=1),
                                       nn.BatchNorm2d(dim),
                                       nn.ReLU(),
                                       WeightNormConv2d(dim, dim, 1))
        else:
            self.block = nn.Sequential(WeightNormConv2d(dim, dim, 3, passinf=1),
                                       nn.BatchNorm2d(dim),
                                       nn.ReLU(),
                                       WeightNormConv2d(dim, dim, 3, padding=1))

    def forward(self, x):
        return x + self.block(self.in_block(x))

class ResModule(nn.Module):
    def __init__(self, in_dim, dim, out_dim, res_blocks, bottleneck):
        '''
        :param in_dim: Number of input features
        :param dim: Number of features in residual blocks
        :param out_dim: Number of output features (should be double of input)
        :param res_blocks: Number of residual blocks
        :param bottleneck: Whether to use bottleneck block or not
        '''
        super(ResModule,self).__init__()
        self.res_blocks = res_blocks
        self.in_block = WeightNormConv2d(in_dim, dim, 3, padding=1)
        self.core_block = nn.ModuleList([ResidualBlock(dim, bottleneck) for _ in range(res_blocks)])
        self.out_block = nn.Sequential(nn.BatchNorm2d(dim),
                                       nn.ReLU(),
                                       WeightNormConv2d(dim, out_dim, 1))

    def forward(self, x):
        x = self.in_block(x)
        for block in self.core_block:
            x = block(x)
        x = self.out_block(x)
        return x


class AffineCheckerboardTransform(nn.Module):
    def __init__(self, in_dim, dim, out_dim, res_blocks, bottleneck, size, config):
        '''
        :param size: mask height/width
        :param config: 1 for first position, 0 for first position
        '''
        super(AffineCheckerboardTransform, self).__init__()
        self.mask = self.create_mask(size, config)
        self.scale_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.shift_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.net = ResModule(in_dim=in_dim, dim=dim, out_dim=out_dim, res_blocks=res_blocks, bottleneck=bottleneck)

    def create_mask(self, size, config):
        mask = (torch.arange(size).view(-1,1) + torch.arange(size))
        if config == 1:
            mask += 1
        return (mask%2).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        self.mask = self.mask.to(x.device)
        x_masked = x * self.mask
        log_scale, shift = self.net(x_masked).chunk(2, dim=1)
        log_scale = log_scale.tanh()*self.scale_scale + self.shift_scale
        log_scale = log_scale*(1 - self.mask)
        shift = shift * (1 - self.mask)
        x = x * log_scale.exp() + shift
        return x, log_scale

class AffineChannelwiseTransform(nn.Module):
    def __init__(self, in_dim, dim, out_dim, res_blocks, bottleneck, top_half_as_input):
        super(AffineChannelwiseTransform, self).__init__()
        self.top_half_as_input = top_half_as_input
        self.scale_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.shift_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.net = ResModule(in_dim=in_dim, dim=dim, out_dim=out_dim, res_blocks=res_blocks, bottleneck=bottleneck)

    def forward(self, x):
        if self.top_half_as_input:
            fixed, not_fixed = x.chunk(2, dim=1)
        else:
            not_fixed, fixed = x.chunk(2, dim=1)
        log_scale, shift = self.net(fixed).chunk(2, dim=1)
        log_scale = log_scale.tanh()*self.scale_scale + self.shift_scale

        if self.top_half_as_input:
            x_modified = torch.cat([fixed, not_fixed], dim=1)
            log_scale = torch.cat([log_scale, torch.zeros_like(log_scale)], dim=1)
        else:
            x_modified = torch.cat([not_fixed, fixed], dim=1)
            log_scale = torch.cat([torch.zeros_like(log_scale), log_scale], dim=1)

        return x_modified, log_scale

class ActNorm(nn.Module):
    def __init__(self, n_channels):
        super(ActNorm, self).__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        self.n_channels = n_channels
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            self.shift.data = -torch.mean(x, dim=[0,2,3], keepdim=True)
            self.log_scale.data = -torch.log(torch.std(x, [0,2,3], keepdim=True))
            self.initialized = True
        return x * torch.exp(self.log_scale) + self.shift, self.log_scale


class RealNVP(nn.Module):
    def __init__(self, in_dim, dim, out_dim, res_blocks, bottleneck, size, type):
        super(RealNVP, self).__init__()
        if type == 'checkerboard':
            self.transforms = nn.ModuleList([AffineCheckerboardTransform(in_dim, dim, out_dim, res_blocks, bottleneck, size, config=1),
                                                       ActNorm(in_dim),
                                                       AffineCheckerboardTransform(in_dim, dim, out_dim, res_blocks, bottleneck, size, config=0),
                                                       ActNorm(in_dim),
                                                       AffineCheckerboardTransform(in_dim, dim, out_dim, res_blocks, bottleneck, size, config=1)])
        elif type == 'channels':
            self.transforms = nn.ModuleList([AffineChannelwiseTransform(in_dim, dim, out_dim, res_blocks, bottleneck, top_half_as_input=True),
                                                     ActNorm(in_dim),
                                                     AffineChannelwiseTransform(in_dim, dim, out_dim, res_blocks, bottleneck, top_half_as_input=True),
                                                     ActNorm(in_dim),
                                                     AffineChannelwiseTransform(in_dim, dim, out_dim, res_blocks, bottleneck, top_half_as_input=True)])
        else:
            print('Error in masking type')

    def forward(self, z):
        z, log_det_J_total = z, torch.zeros_like(z)
        for transform in self.transforms:
            z, log_det = transform(z)
            log_det_J_total += log_det

        return z, log_det_J_total

'''im = torch.rand(2,32,16,16)
res = RealNVP(32, 32, 64, 2, True, 16, 'checkerboard')
z, log_det = res(im)
print(z.shape)
print(log_det.shape)'''

