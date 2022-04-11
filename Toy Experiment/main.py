import torch
import copy
import os
from torch.utils.data import DataLoader

import configuration
import dataset
import utils
import model
import visualization
import algorithms


def experiment(args):
    if args.seed is not None:
        utils.set_random(args.seed)

    #Load GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Create Dataset
    (X_s, y_s, _), (X_t, y_t, _), _ = dataset.sample(rot=args.rot, tran=args.tran, sep=args.sep)
    train_dataset = dataset.ToyDataset(X_s, y_s)
    train_loader = DataLoader(train_dataset, batch_size=args.n, shuffle=True)


    test_dataset = dataset.ToyDataset(X_t, y_t)
    test_loader = DataLoader(test_dataset, batch_size=args.n, shuffle=True)

    #Model
    if args.flow == 'cdf':
        net = model.net_cdf(n_components=3, norm=True).to(device)
        add = '_cdf'
    if args.flow == 'affine':
        #net = model2.net_affine(norm=True).to(device)
        net = model.net_affine(b_size=X_s.shape[0], norm=True).to(device)
        add = '_affine'

    #Training
    acc, train_loss, train_cls_loss, train_uns_loss = algorithms.train(device, net, train_loader, args)
    utils.plot_loss(train_cls_loss, 'Training Classification loss', 'plots/train_cls_loss' + add + '.png')
    utils.plot_loss(train_uns_loss, 'Training Unsupervised loss', 'plots/train_uns_loss' + add + '.png')
    utils.plot_loss(train_loss, 'Training Total loss', 'plots/train_loss' + add + '.png')
    utils.plot_loss(acc, 'Training Accuracy', 'plots/train_acc' + add + '.png')

    #Testing
    if args.flow == 'cdf':
        net = model.net_cdf(n_components=3, freeze=True, norm=True).to(device)
        checkpoint = torch.load('results_cdf.pt')
    if args.flow == 'affine':
        #net = model2.net_affine(norm=True).to(device)
        net = model.net_affine(b_size=X_s.shape[0], freeze=True, norm=True).to(device)
        checkpoint = torch.load('results_affine.pt')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    test_acc = algorithms.test(device, net, test_loader)
    print('Test accuracy after training:', test_acc)

    visualization.plot_prediction(X_s, y_s, copy.deepcopy(net).cpu(), 3, 'plots/DecisionBoundary/Training/db' + add + '.png')
    visualization.plot_prediction(X_t, y_t, copy.deepcopy(net).cpu(), 3, 'plots/DecisionBoundary/Testing/db' + add + '.png')

    #Test Time Adaptation
    acc, loss, net_bkp = algorithms.adapt(device, net, test_loader, args)
    utils.plot_loss(loss, 'Testing Unsupervised loss', 'plots/test_uns_loss' + add + '.png')
    utils.plot_loss(acc, 'Testing Accuracy', 'plots/train_acc' + add + '.png')

    net.load_state_dict(net_bkp)
    visualization.plot_prediction(X_t, y_t, copy.deepcopy(net).cpu(), 3,
                                   'plots/DecisionBoundary/Testing/Final_db' + add + '.png')


if __name__ == '__main__':
    args = configuration.argparser()
    experiment(args)