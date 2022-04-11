import torch
import torch.nn as nn
from torch.distributions.normal import Normal

'''Flow 1D
    We choose the Normal Distribution Cumulative Density Function (CDF) as the bijective transformation. The derivative
    of a CDF is the PDF, thus being easy to compute.
             _____
    x  ---> | CDF | ---> z   
             ¯¯¯¯¯
    More specifically, the transformation we use is a linear combination of n CDFs:

    z = w1*CDF1 + w2*CDF2 + ... + wn*CDFn, where w1 + w2 + ... + wn = 1
'''


class Flow1d(nn.Module):
    def __init__(self, n_components):
        super(Flow1d, self).__init__()
        self.mus = nn.Parameter(torch.randn(n_components), requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        self.weight_logits = nn.Parameter(torch.ones(n_components), requires_grad=True)

    def forward(self, x):
        x = x.view(-1, 1)
        weights = self.weight_logits.softmax(dim=0).view(1, -1)
        distribution = Normal(self.mus, self.log_sigmas.exp())
        z = (distribution.cdf(x) * weights).sum(dim=1)
        dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(dim=1)
        return z, dz_by_dx


'''CDF Parameter generator
    Based on one of the variables in the input, this function generates the 
    parameters of the transformation for the other variable.
            ___
    x ---> | m | ---> theta
            ¯¯¯
    In this case, theta comprises Mus, Sigmas, and mixture weights for the n different CDFs
'''


class CDFParams(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_hidden_layers=3, output_size=None):
        super(CDFParams, self).__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


'''Conditional Flow
    A 1D Flow maps x2 to z2, conditioned by x1. This means that x1 is used to generate the 
    parameters of the Flow that transforms x2.
              _____
    x2  ---> | CDF | ---> z2
           __^¯¯¯¯¯
        __|
       |        
    x1  ----------------> z1            
'''


class ConditionalFlow1D(nn.Module):
    def __init__(self, n_components):
        super(ConditionalFlow1D, self).__init__()
        self.cdf = CDFParams(output_size=n_components)

    def forward(self, x, condition):  # Condition is x1
        x = x.view(-1, 1)
        mus, log_sigmas, weight_logits = torch.chunk(self.cdf(condition), 3, dim=1)
        weights = weight_logits.softmax(dim=1)
        distribution = Normal(mus, log_sigmas.exp())
        z = (distribution.cdf(x) * weights).sum(dim=1)
        dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(dim=1)
        return z, dz_by_dx


'''2D Flow Model
    The model that combines a mixture of CDFs as the transformation for the first dimension, 
    and a CDF parameter generator for the second dimension
'''


class Flow2d(nn.Module):
    def __init__(self, n_components):
        super(Flow2d, self).__init__()
        self.flow_dim1 = Flow1d(n_components)
        self.flow_dim2 = ConditionalFlow1D(n_components)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        z1, dz1_by_dx1 = self.flow_dim1(x1)
        z2, dz2_by_dx2 = self.flow_dim2(x2, condition=x1)
        z = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
        dz_by_dx = torch.cat([dz1_by_dx1.unsqueeze(1), dz2_by_dx2.unsqueeze(1)], dim=1)

        return z, dz_by_dx