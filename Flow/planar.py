import torch
import torch.nn as nn

'''Planar Flow
Computes the following transformation:
            
            f(x) = x + u * h(w*x + b)
            
where u, w and b are trainable parameters, and h is an activation function
'''
class PlanarTransform(nn.Module):
    def __init__(self, shape):
        '''
        :param shape: shape of the input (1 x dim)
        '''
        super().__init__()
        self.w = nn.Parameter(torch.randn(1,shape).normal_(0, 0.1))
        self.b = nn.Parameter(torch.randn(1,shape).normal_(0, 0.1))
        self.u = nn.Parameter(torch.randn(1,shape).normal_(0, 0.1))
        self.h = torch.tanh

    def forward(self, x):
        if torch.mm(self.u, self.w.T) < -1:
            self.get_u_hat() #Conditioning u to satisfy w*u >= - 1

        affine = torch.mm(x, self.w.T) + self.b
        #Computing transformation
        z = self.u * self.h(affine)

        #Computing log of Jacobian determinant
        derivative = (1 - self.h(affine)**2)*self.w
        abs_det = (1 + torch.mm(self.u, derivative.T)).abs()
        log_det = torch.log(1e-4 + abs_det)

        return z, log_det

    def get_u_hat(self):
        wtu = torch.mm(self.u, self.w.T)
        m_wtu = -1 + torch.log(1 + torch.exp(wtu))
        self.u.data = (self.u + (m_wtu - wtu) * self.w / torch.norm(self.w, p=2, dim=1) ** 2)

class PlanarFlow(nn.Module):
    def __init__(self, shape, K = 6):
        '''
        :param shape:  shape of the input (1 x dim)
        :K: number of transformations to compose
        '''
        super().__init__()
        self.layers = [PlanarTransform(shape) for _ in range(K)]
        self.model = nn.Sequential(*self.layers)

    def forward(self, z):
        log_det_J = 0
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_J += log_det

        return z, log_det_J

print(x.shape)
flow = PlanarFlow(48)
z, log_det = flow(x)
print(z.shape)