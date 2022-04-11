import torch
from torch.utils.data import DataLoader

import configuration
import dataset
import utils
import model
import model2
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
    if args.flow == 'affine':
        #net = model2.net_affine(norm=True).to(device)
        net = model.net_affine(b_size=X_s.shape[0], norm=True).to(device)

    #Training
    acc, train_loss, train_cls_loss, train_uns_loss = algorithms.train(device, net, train_loader, args)
    print(acc.values)
    utils.plot_loss(train_cls_loss, 'Training Classification loss', 'plots/train_cls_loss.png')
    utils.plot_loss(train_uns_loss, 'Training Unsupervised loss', 'plots/train_uns_loss.png')
    utils.plot_loss(train_loss, 'Training Total loss', 'plots/train_loss.png')
    utils.plot_loss(acc, 'Training Accuracy', 'plots/train_acc.png')

    #Testing


if __name__ == '__main__':
    args = configuration.argparser()
    experiment(args)