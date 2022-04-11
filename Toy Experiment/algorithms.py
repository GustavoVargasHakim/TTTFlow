import copy
import torch
import torch.nn as nn
from tqdm import tqdm

import utils
import visualization

def train(device, net, dataloader, args):
    if args.train_opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_train)
    elif args.train_opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr_train)

    #Loss functions
    if args.flow == 'cdf':
        criterion_uns = utils.cdf_likelihood
        add = '_cdf'
    elif args.flow == 'affine':
        criterion_uns = utils.affine_likelihood
        add = '_affine'
    criterion_cls = nn.BCEWithLogitsLoss()
    beta = args.beta

    #Training loop
    acc = utils.AverageMeter(device, args.epochs)
    loss_cls_avg = utils.AverageMeter(device, args.epochs)
    loss_uns_avg = utils.AverageMeter(device, args.epochs)
    loss_avg = utils.AverageMeter(device, args.epochs)
    net.train()
    for epoch in tqdm(range(args.epochs)):
        for i, data in enumerate(dataloader, 0):
            x = data['input'].to(device, non_blocking=True)
            x = torch.nn.functional.normalize(x)
            y = data['label'].to(device, non_blocking=True)

            out, (z, dz_by_dx) = net(x)

            loss_cls = criterion_cls(out, y.unsqueeze(1))
            loss_uns = criterion_uns(device, z, dz_by_dx)
            loss = loss_cls + beta*loss_uns

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        acc.append(utils.accuracy(out, y))
        loss_cls_avg.append(loss_cls)
        loss_uns_avg.append(loss_uns)
        loss_avg.append(loss)

        if epoch % args.display_freq == 0 :
            tqdm.write('Classification loss: {}'.format(loss_cls_avg.avg))
            tqdm.write('Unsupervised loss: {}'.format(loss_uns_avg.avg))
            tqdm.write('Total loss: {}'.format(loss_avg.avg))
            tqdm.write('Accuracy: {}'.format(acc.avg))

        #fig = visualization.plot_prediction(X, Y, copy.deepcopy(net).cpu(), scale=3)
        #fig.savefig('plots/DecisionBoundary/Training/db' + add + str(epoch) + '.png')

    return acc, loss_avg, loss_cls_avg, loss_uns_avg


