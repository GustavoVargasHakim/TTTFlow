import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt

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

def set_random(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cdf_likelihood(device, z, dz_by_dx):
    target_distribution = Normal(torch.tensor([0.0, 0.0], device=device), torch.tensor([1.0, 1.0], device=device))
    log_likelihood_per_dim = target_distribution.log_prob(z) + dz_by_dx.log()
    log_likelihood = log_likelihood_per_dim.sum(1)
    return -log_likelihood.mean()

def affine_likelihood(device, z, dz_by_dx):
    target_distribution = Normal(torch.tensor([0.0, 0.0], device=device), torch.tensor([1.0, 1.0], device=device))
    log_likelihood_per_dim = target_distribution.log_prob(z) + dz_by_dx
    log_likelihood = log_likelihood_per_dim.sum(1)
    return -log_likelihood.mean()

def accuracy(output, target):
    return (output.squeeze().gt(0.0) == target).sum()*100 / target.size(0)

def plot_loss(loss, name, figdir):
    plt.figure()
    plt.plot(loss.values_list)
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(figdir, bbox_inches='tight')
    plt.show()
    plt.close()



