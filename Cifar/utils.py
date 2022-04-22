import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions import MultivariateNormal
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

def det_likelihood(device, z, log_det):
    target_distribution = Normal(torch.zeros(z.size()[1], device=device), torch.ones(z.size()[1], device=device))
    # target_distribution = MultivariateNormal(torch.zeros(z.shape[1], device=device), torch.eye(z.shape[1], device=device))
    log_likelihood_per_dim = target_distribution.log_prob(z) + log_det
    log_likelihood = log_likelihood_per_dim.sum(1)
    return -log_likelihood.mean()

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    #Discomment the following line if 'target' is a batch of one_hot vectors
    #classes = (target == 1).nonzero(as_tuple=True)[1]#.view(-1,1).t()
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, dim=1)
        pred = pred.t()
        correct = (pred == target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)[0]
        res = (correct_k.mul_(100.0 / batch_size))
        return res

def plot_loss(loss, name, figdir):
    plt.figure()
    plt.plot(loss.values_list)
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(figdir, bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_losses(train, val, name, figdir):
    plt.figure()
    plt.plot(train.values_list, label = 'Train')
    plt.plot(val.values_list, label = 'Valid')
    plt.legend(('Train', 'Validation'))
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(figdir, bbox_inches='tight')
    #plt.show()
    plt.close()