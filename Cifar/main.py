import torch
import copy
import os
from torch.utils.data import DataLoader, random_split

import configuration
import dataset
import utils
import model
import visualization
import algorithms
import models.ResNetTTAFlow as ResNetTTAFlow

def experiment(args):
    if args.seed is not None:
        utils.set_random(args.seed)

    #Load GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Create Dataset
    # X_s_train, y_s_train = dataset.read_Cifar('C:/Users/dosowiec/Documents/Data/cifar-100-python/train')
    # X_s_test, y_s_test = dataset.read_Cifar('C:/Users/dosowiec/Documents/Data/cifar-100-python/test')
    # train_s_dataset = dataset.ToyDataset(X_s_train, y_s_train)
    # val_size=5000
    # train_size = len(train_s_dataset) - val_size
    # train_s_dataset, val_s_dataset = random_split(train_s_dataset, [train_size, val_size])
    # train_s_loader = DataLoader(train_s_dataset, batch_size=args.n, shuffle=True)
    # val_s_loader = DataLoader(val_s_dataset, batch_size=args.n, shuffle=False)

    _, val_s_loader = dataset.read_Cifar_test_ttt('C:/Users/dosowiec/Documents/Data', args)
    _, train_s_loader = dataset.read_Cifar_train_ttt('C:/Users/dosowiec/Documents/Data', args)

    _, test_t_loader = dataset.read_Cifar_C('C:/Users/dosowiec/Documents/Data/CIFAR-10-C/brightness.npy', 'C:/Users/dosowiec/Documents/Data', args)

    # test_s_dataset = dataset.ToyDataset(X_s_test, y_s_test)
    # test_s_loader = DataLoader(test_s_dataset, batch_size=args.n, shuffle=True)

    #Model
    # net, ext, head, uns = model.build_model(args, device)
    depth = 26
    width = 1
    shape = 64*width
    net = ResNetTTAFlow.ResNetCifar(depth, 4096, args.n).to(device)

    #Training
    acc_train, train_cls_loss, train_uns_loss, train_loss, acc_val, val_cls_loss, val_uns_loss, val_loss = algorithms.train(device, net, train_s_loader, val_s_loader, args)
    utils.plot_losses(train_cls_loss, val_cls_loss, 'Training Classification loss', 'plots/train_cls_loss.png')
    utils.plot_losses(train_uns_loss, val_uns_loss, 'Training Unsupervised loss', 'plots/train_uns_loss.png')
    utils.plot_losses(train_loss, val_loss, 'Training Total loss', 'plots/train_loss.png')
    utils.plot_losses(acc_train, acc_val, 'Training Accuracy', 'plots/train_acc.png')

    #Testing
    net = ResNetTTAFlow.ResNetCifar(depth, 4096, freeze=True).to(device)
    checkpoint = torch.load('results.pt')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    test_acc = algorithms.test(device, net, test_t_loader)
    print('Test accuracy after training:', test_acc)

    #Test Time Adaptation
    test_acc, test_loss = algorithms.adapt(device, net, test_t_loader, args)
    utils.plot_loss(test_loss,  'Training Total loss', 'plots/test_loss.png')
    utils.plot_loss(test_acc, 'Training Accuracy', 'plots/test_acc.png')



if __name__ == '__main__':
    args = configuration.argparser()
    experiment(args)