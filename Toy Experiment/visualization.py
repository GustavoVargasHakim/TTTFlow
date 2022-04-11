import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def plot_prediction(x, y, net, scale):
    """plot decision boundary"""
    net.test = True
    fig = plt.figure()
    fig.set_size_inches(2, 2)
    cm_bright = colors.ListedColormap(['#FF0000', '#0000FF'])
    x0s = np.linspace(-scale, scale, 1000)
    x1s = np.linspace(-scale, scale, 1000)
    x0, x1 = np.meshgrid(x0s, x1s)
    xe = np.c_[x0.ravel(), x1.ravel()]
    xe = torch.from_numpy(xe).float()
    net.eval()
    with torch.no_grad():
        y_pred = net(xe)
    '''y_pred = y_pred.squeeze().gt(0.0)
    y_pred = y_pred.reshape(x0.shape).detach().numpy()'''
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = np.array(y_pred).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=cm_bright, alpha=0.05)
    plt.scatter(x[:, 0], x[:, 1], s=2.0, c=y, alpha=0.5, cmap=cm_bright)
    plt.axis('square')
    plt.xlim([-scale, scale])
    plt.ylim([-scale, scale])
    #plt.plot()
    return fig