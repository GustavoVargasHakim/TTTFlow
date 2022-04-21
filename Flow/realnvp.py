import torch
import torch.nn as nn
import numpy as np

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
    def __init__(self, in_dim, dim, out_dim, res_blocks, bottleneck, skip=True):
        '''
        :param in_dim: Number of input features
        :param dim: Number of features in residual blocks
        :param out_dim: Number of output features (should be double of input)
        :param res_blocks: Number of residual blocks
        :param bottleneck: Whether to use bottleneck block or not
        :param skip: Whether to use skip connections or not
        '''
        super(ResModule,self).__init__()
        self.res_blocks = res_blocks
        self.skip = skip
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
        los_scale = log_scale.tanh()*self.scale_scale + self.shift_scale
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
            results = x*torch.exp(self.log_scale) + self.shift
        return x * torch.exp(self.log_scale) + self.shift, self.log_scale


class RealNVP(nn.Module):
    def __init__(self, in_dim, dim, out_dim, res_blocks, bottleneck, size, type):
        super(RealNVP, self).__init__()
        if type == 'checkerboard':
            self.transforms = nn.ModuleList([AffineCheckerboardTransform(in_dim, dim, out_dim, res_blocks, bottleneck, size, config=1),
                                                       ActNorm(out_dim),
                                                       AffineCheckerboardTransform(in_dim, dim, out_dim, res_blocks, bottleneck, size, config=0),
                                                       ActNorm(out_dim),
                                                       AffineCheckerboardTransform(in_dim, dim, out_dim, res_blocks, bottleneck, size, config=1)])
        elif type == 'channels':
            self.transforms = nn.ModuleList([AffineChannelwiseTransform(in_dim, dim, out_dim, res_blocks, bottleneck, top_half_as_input=True),
                                                     ActNorm(out_dim),
                                                     AffineChannelwiseTransform(in_dim, dim, out_dim, res_blocks, bottleneck, top_half_as_input=True),
                                                     ActNorm(out_dim),
                                                     AffineChannelwiseTransform(in_dim, dim, out_dim, res_blocks, bottleneck, top_half_as_input=True)])
        else:
            print('Error in masking type')

    def forward(self, x):
        z, log_det_J = x, torch.zeros_like(x)

        for transform in self.transforms:
            z, log_det = transform(z)
            log_det_J += log_det

        return z, log_det_J

im = torch.rand(2,3,32,32)
res = RealNVP(3, 3, 6, 2, True, 32, 'checkerboard')
z, log_det = res(im)

'''class AbstractCoupling(nn.Module):
    def __init__(self, mask_config, res_blocks, bottleneck, skip):
        :param mask_config: masking configuration
        :param res_blocks: Number of resblocks
        :param bottleneck: Whether to use bottleneck resblocks or not
        :param skip: Whether to use skip connections or not
        self.mask_config = mask_config
        self.res_blcoks = res_blocks
        self.bottleneck = bottleneck
        self.skip = skip

    def build_mask(self, size, config=1.):
        Only for checkerboards mask
        mask = np.arange(size).reshape(-1,1) + np.arange(size)
        mask = np.mod(config + mask, 2)
        mask = mask.reshape(-1,1, size, size)
        return torch.tensor(mask.astype('float32'))

    def batch_stat(self, x):
        Computing spatiat batch statistics
        mean = torch.mean(x, dim=(0,2,3), keepdim=True)
        var = torch.mean((x-mean)**2, dim=(0,2,3), keepdim=True)
        return mean, var

class CheckerboardAdditiveCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, mid_dim, size, res_blocks, bottleneck, skip, mask_config, coupling_bn):
        :param in_out_dim: Number of input and output features
        :param mid_dim: Number of features in residual blocks
        :param size: Height/width of features
        :param res_blocks: Number of residual blocks
        :param bottleneck: Whether to use the bottleneck block or not
        :param skip: Whether to use the skip connection or not
        :param mask_config: Mask configuration
        :param coupling_bn:
        super(CheckerboardAdditiveCoupling, self).__init__()
        self.coupling_bn = coupling_bn
        self.mask = self.build_mask(size, config=mask_config).cuda()
        self.in_bn = nn.BatchNorm2d(in_out_dim)
        self.block = nn.Sequential(nn.ReLU(),
                                   ResModule(2*in_out_dim+1, mid_dim, in_out_dim, res_blocks, bottleneck, skip))
        self.out_bn = nn.BatchnOrm2d(in_out_dim, affine=False)

    def forward(self, x):
        [B, _, _, _] = x.shape()
        mask = self.mask.repeat(B,1,1,1)
        x_ = self.in_bn(x*mask)
        x_ = torch.cat((x_, -x_), dim=1)
        x_ = torch.cat((x_, mask), dim=1)
        shift = self.block(x_)*(1. - mask)

        log_diag_J = torch.zeros_like(x)
        x = x + shift
        if self.coupling_bn:
            if self.training:
                _, var = self.batch_stat(x)
            else:
                var = self.out_bn.running_var
                var = var.reshape(-1,1,1,1).transpose(0,1)
            x = self.out_bn(x)*(1.-mask) + x*mask
            log_diag_J = log_diag_J - 0.5*torch.log(var + 1e-5)*(1.-mask)
        return x, log_diag_J

class CheckerboardAffineCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, mid_dim, size, res_blocks, bottleneck, skip, mask_config, coupling_bn):
        :param in_out_dim: Number of input and output features
        :param mid_dim: Number of features in residual blocks
        :param size: Height/width of features
        :param res_blocks: Number of residual blocks
        :param bottleneck: Whether to use the bottleneck block or not
        :param skip: Whether to use the skip connection or not
        :param mask_config: Mask configuration
        :param coupling_bn: 
        super(CheckerboardAdditiveCoupling, self).__init__()
        self.mask = self.build_mask(size, config=mask_config).cuda()
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Paraemter(torch.zeros(1), requires_grad=True)
        self.in_bn = nn.BatchNorm2d(in_out_dim)
        self.block = nn.Sequential(nn.ReLU(),
                                   ResModule(2*in_out_dim+1, mid_dim, 2*in_out_dim, res_blocks, bottleneck, skip=True))
        self.out_bn = nn.BatchNorm2d(in_out_dim, affine=False)

    def forward(self, x):
        [B, C, _, _] = list(x.size())
        mask = self.mask.repeat(B, 1, 1, 1)
        x_ = self.in_bn(x * mask)
        x_ = torch.cat((x_, -x_), dim=1)
        x_ = torch.cat((x_, mask), dim=1)  # 2C+1 channels
        (shift, log_rescale) = self.block(x_).split(C, dim=1)
        log_rescale = self.scale * torch.tanh(log_rescale) + self.scale_shift
        shift = shift * (1. - mask)
        log_rescale = log_rescale * (1. - mask)

        log_diag_J = log_rescale
        x = x*torch.exp(log_rescale) + shift
        if self.coupling_bn:
            if self.training:
                _, var = self.batch_stat(x)
            else:
                var = self.out_bn.running_var
                var = var.reshape(-1,1,1,1).transpose(0,1)
            x = self.out_bn(x)*(1. - mask) + x*mask
            log_diag_J = log_diag_J - 0.5*torch.log(var + 1e-5)+(1. - mask)
        return x, log_diag_J'''

