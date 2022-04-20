import torch
import torch.nn as nn

from splits import Split, Squeeze, Merge

class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels, use_lu=False):
        '''
        :param num_channels: Number of channels of input
        :param use_lu: Use the LU decomposition (bool)
        '''
        super().__init__()
        self.num_channels = num_channels
        self.use_lu = use_lu
        Q = torch.linalg.qr(torch.randn(self.num_channels, self.num_channels))[0]
        if use_lu:
            P, L, U = torch.lu_unpack(*Q.lu())
            self.register_buffer('P',P)
            self.L = nn.Parameter(L) #Lower triangular matrix
            S = U.diag()
            self.register_buffer('sign_S', torch.sign(S))
            self.log_S = nn.Parameter(torch.log(torch.abs(S)))
            self.U = nn.Parameter(torch.triu(U, diagonal=1))
            self.register_buffer('eye', torch.diag(torch.ones(self.num_channels)))
        else:
            self.W = nn.Parameter(Q)

    def forward(self, x):
        if self.use_lu:
            W = self._assemble_W(inverse=True)
            log_det = -torch.sum(self.logS)
        else:
            W_dtype = self.W.dtype
            if W_dtype == torch.float64:
                W = torch.inverse(self.W)
            else:
                W = torch.inverse(self.W.double()).type(W_dtype)
            W = W.view(*W.size(), 1, 1)
            log_det = -torch.slogdet(self.W)[1]
        W = W.view(self.num_channels, self.num_channels, 1,1)
        z_ = torch.functional.conv2d(x, W)
        log_det = log_det*z.size(2)*z.size(3)
        return z_, log_det

    def _assemble_W(self, inverse=False):
        # assemble W from its components (P, L, U, S)
        L = torch.tril(self.L, diagonal=-1) + self.eye
        U = torch.triu(self.U, diagonal=1) + torch.diag(self.sign_S * torch.exp(self.log_S))
        if inverse:
            if self.log_S.dtype == torch.float64:
                L_inv = torch.inverse(L)
                U_inv = torch.inverse(U)
            else:
                L_inv = torch.inverse(L.double()).type(self.log_S.dtype)
                U_inv = torch.inverse(U.double()).type(self.log_S.dtype)
            W = U_inv @ L_inv @ self.P.t()
        else:
            W = self.P @ L @ U
        return W

class AffineCoupling(nn.Module):
    def __init__(self, param_map, scale=True, scale_map='exp'):
        '''
        :param param_map: Mapping features to shift and scale parameters
        :param scale: To apply scaling or not (bool)
        :param scale_map: Map to be applied to the scale parameter, 'exp' for RealNVP, 'sigmoid' for Glow
        '''
        super().__init__()
        self.add_module('param_map', param_map)
        self.scale = scale
        self.scale_map = scale_map

    def forward(self,z):
        '''
        z is a list of z1 and z2; z = [z1, z2]
        z1 is left constant and affine map is applied to z2 with parameters depending on z1
        '''
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == 'exp':
                z2 = z2 * torch.exp(scale_) + shift
                log_det = torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid':
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 / scale + shift
                log_det = -torch.sum(torch.log(scale),
                    dim=list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid_inv':
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 * scale + shift
                log_det = torch.sum(torch.log(scale),
                    dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError('This scale map is not implemented.')
        else:
            z2 += param
            log_det = 0
        return [z1, z2], log_det

class AffineCouplingBlock(nn.Module):
    def __init__(self, param_map, scale=True, scale_map='exp', split_mode='channel'):
        '''
        :param param_map: Mapping features to shift and scale parameter
        :param scale: Applying scale or not (bool)
        :param scale_map:  Map to be applied to the scale parameter, 'exp' for RealNVP and 'sigmoid' for Glow
        :param split_mode: Splitting mode
        '''
        super().__init__()
        self.flows = nn.ModuleList([])
        self.flows += [Split(split_mode)]
        self.flows += [AffineCoupling(param_map, scale, scale_map)]
        self.flows += [Merge(split_mode)]

    def forward(self,z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

class AffineConstFlow(nn.Module):
    def __init__(self, shape, scale=True, shift=True):
        """
        Constructor
        :param shape: Shape of the coupling layer
        :param scale: Flag whether to apply scaling
        :param shift: Flag whether to apply shift
        :param logscale_factor: Optional factor which can be used to control
        the scale of the log scale factor
        """
        super().__init__()
        if scale:
            self.s = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer('s', torch.zeros(shape)[None])
        if shift:
            self.t = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer('t', torch.zeros(shape)[None])
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(torch.tensor(self.s.shape) == 1, as_tuple=False)[:, 0].tolist()

    def forward(self, z):
        z_ = z * torch.exp(self.s) + self.t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(self.s)
        return z_, log_det

class ActNorm(AffineConstFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done_cpu = torch.tensor(0.)
        self.register_buffer('data_dep_init_done', self.data_dep_init_done_cpu)

    def forward(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done > 0.:
            assert self.s is not None and self.t is not None
            s_init = -torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = (-z.mean(dim=self.batch_dims, keepdim=True) * torch.exp(self.s)).data
            self.data_dep_init_done = torch.tensor(1.)
        return super().forward(z)

class ConvNet2d(nn.Module):
    """
    Convolutional Neural Network with leaky ReLU nonlinearities
    """

    def __init__(self, channels, kernel_size, leaky=0.0, init_zeros=True,
                 actnorm=False, weight_std=None):
        """
        Constructor
        :param channels: List of channels of conv layers, first entry is in_channels
        :param kernel_size: List of kernel sizes, same for height and width
        :param leaky: Leaky part of ReLU
        :param init_zeros: Flag whether last layer shall be initialized with zeros
        :param scale_output: Flag whether to scale output with a log scale parameter
        :param logscale_factor: Constant factor to be multiplied to log scaling
        :param actnorm: Flag whether activation normalization shall be done after
        each conv layer except output
        :param weight_std: Fixed std used to initialize every layer
        """
        super().__init__()
        # Build network
        net = nn.ModuleList([])
        for i in range(len(kernel_size) - 1):
            conv = nn.Conv2d(channels[i], channels[i + 1], kernel_size[i],
                             padding=kernel_size[i] // 2, bias=(not actnorm))
            if weight_std is not None:
                conv.weight.data.normal_(mean=0.0, std=weight_std)
            net.append(conv)
            if actnorm:
                net.append(utils.ActNorm((channels[i + 1],) + (1, 1)))
            net.append(nn.LeakyReLU(leaky))
        i = len(kernel_size)
        net.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size[i - 1],
                             padding=kernel_size[i - 1] // 2))
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class GlowBlock(nn.Module):
    def __init__(self, channels, hidden_channels, scale=True, scale_map='sigmoid',
                 split_mode='channel', leaky=0.0, init_zeros=True, use_lu=True,
                 net_actnorm=False):
        '''
        :param channels: Number of channels of the input
        :param hidden_channels: Number of channels in the hidden layer of the ConvNet
        :param scale: To use scaling or not in affine coupling layer (bool)
        :param scale_map: Map to be applied to the scale parameters
        :param split_mode: Splitting method
        :param leaky: Leaky parameter of LeakyReLU of ConvNet2d
        :param init_zeros: To initialize last conv layer with zeros or not (bool)
        :param use_lu: To parametrize convolutional weights through LU decomposition in 1x1 conv layers or not (bool)
        '''
        super().__init__()
        self.flows = nn.ModuleList([])
        kernel_size = (3,1,3)
        num_param = 2 if scale else 1
        if 'channel' == split_mode:
            channels_ = (channels // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * ((channels + 1) // 2),)
        elif 'channel_inv' == split_mode:
            channels_ = ((channels + 1) // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * (channels // 2),)
        elif 'checkerboard' in split_mode:
            channels_ = (channels,) + 2 * (hidden_channels,)
            channels_ += (num_param * channels,)
        else:
            raise NotImplementedError('Mode ' + split_mode + ' is not implemented.')
        param_map = ConvNet2d(channels_, kernel_size, leaky, init_zeros,
            actnorm=net_actnorm)
        self.flows += [AffineCouplingBlock(param_map, scale, scale_map, split_mode)]
        # Invertible 1x1 convolution
        if channels > 1:
            self.flows += [Invertible1x1Conv(channels, use_lu)]
        # Activation normalization
        self.flows += [ActNorm((channels,) + (1, 1))]

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

im = torch.rand(1,2,10,10)
flow = GlowBlock(3,2)
z, log_det = flow(im)