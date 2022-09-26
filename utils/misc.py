import os
import torch
import torch.nn as nn
import torch.nn.functional as F
#from colorama import Fore

def get_flow_weights(args):
	if args.shared == 'layer2':
		return '/home/vhakim/projects/rrg-ebrahimi/vhakim/TTTCifar/Results/flow_layer2_0.001_CIFAR10.pth'
	elif args.shared == 'layer3':
		return '/home/vhakim/projects/rrg-ebrahimi/vhakim/TTTCifar/Results/flow_layer3_0001.pth'


def my_makedir(name):
	try:
		os.makedirs(name)
	except OSError:
		pass

def mean(ls):
	return sum(ls) / len(ls)

def flat_grad(grad_tuple):
    return torch.cat([p.view(-1) for p in grad_tuple])

def neg_log_likelihood(target, z, log_det):
	log_likelihood_per_dim = target.log_prob(z) + log_det
	return -log_likelihood_per_dim.sum(1).mean()

def neg_log_likelihood_2d(target, z, log_det):
	log_likelihood_per_dim = target.log_prob(z) + log_det
	return -log_likelihood_per_dim.mean()

def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                if isinstance(child, nn.Conv2d):
                    child.weight.requires_grad = False
                flatt_children.extend(get_children(child))
            except TypeError:
                if isinstance(child, nn.Conv2d):
                    child.weight.requires_grad = False
                flatt_children.append(get_children(child))


def configure_model(model):
   """Configure model for use with tent."""
   # train mode, because tent optimizes the model to minimize entropy
   model.train()
   # disable grad, to (re-)enable only what tent updates
   model.requires_grad_(False)
   # configure norm for tent updates: enable grad + force batch statisics
   for m in model.modules():
      if isinstance(m, nn.BatchNorm2d):
         m.requires_grad_(True)
         # force use of batch stats in train and eval modes
         m.track_running_stats = False
         m.running_mean = None
         m.running_var = None
   return model

def collect_params(model):
   """Collect the affine scale + shift parameters from batch norms.
       Walk the model's modules and collect all batch normalization parameters.
       Return the parameters and their names.
       Note: other choices of parameterization are possible!
       """
   params = []
   names = []
   for nm, m in model.named_modules():
      if isinstance(m, nn.BatchNorm2d):
         for np, p in m.named_parameters():
            if np in ['weight', 'bias']:  # weight is scale, bias is shift
               params.append(p)
               names.append(f"{nm}.{np}")
   return params, names

class AverageMeter:
    """Computes and stores the average and current value on device"""
    def __init__(self, device, length):
        self.device = device
        self.length = length
        self.reset()

    def reset(self):
        self.values = torch.zeros(self.length, device=self.device, dtype=torch.float)
        self.counter = 0
        self.last_counter = 0

    def append(self, val):
        self.values[self.counter] = val.detach()
        self.counter += 1
        self.last_counter += 1

    @property
    def val(self):
        return self.values[self.counter - 1]

    @property
    def avg(self):
        return self.values[:self.counter].mean()

    @property
    def values_list(self):
        return self.values[:self.counter].cpu().tolist()

    @property
    def last_avg(self):
        if self.last_counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = self.values[self.counter - self.last_counter:self.counter].mean()
            self.last_counter = 0
            return self.latest_avg

def get_message(setting):
    if setting == 'pre+flow':
        print('Train the Flow on top of pre-trained ResNet with the unsupervised loss')
    if setting == 'classification':
        print('Only training ResNet for accuracy and calibration')

def get_entropy(x):
    return -(x*x.log()).sum()

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def patch(x, patch_size, stride=[1,1], padding=[2,2]):
    ph, pw = padding
    sh, sw = stride
    h, w = patch_size
    h = (h)//2
    w = (w)//2
    H, W = x.shape[-2:]
    [X, Y, dX, dY] = torch.meshgrid(torch.arange(0+(h-ph), H-(h-ph), step=sh),
                                    torch.arange(0+(w-pw), W-(w-pw), step=sw),
                                    torch.arange(-h, h+1), torch.arange(-w, w+1))
    X = X + dX
    Y = Y + dY
    X[X < 0] = 2 - X[X < 0]
    Y[Y < 0] = 2 - Y[Y < 0]
    X[X >= H] = 2*(H-1)-X[X >= H]
    Y[Y >= W] = 2*(W-1)-Y[Y >= W]
    return x[...,X, Y]

def get_patches(x, size):
    N, C, H, W = x.shape[0:4]
    patches = x.unfold(2,size,size).unfold(3,size,size).permute(0,2,3,1,4,5).reshape(N*(H//size)*(W//size),C,size,size)
    return patches
