import torch
import torch.nn as nn
from tqdm import tqdm
import copy

import utils
import visualization


def train(device, net, dataloader_train, dataloader_valid, args):
    if args.train_opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_train)
    elif args.train_opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr_train)

    #Loss functions
    #criterion_uns = utils.cdf_likelihood
    criterion_uns = utils.det_likelihood
    criterion_cls = nn.CrossEntropyLoss()
    beta = args.beta

    #Training stats
    acc_best = 0
    acc_train = utils.AverageMeter(device, args.epochs)
    loss_cls_avg_train = utils.AverageMeter(device, args.epochs)
    loss_uns_avg_train = utils.AverageMeter(device, args.epochs)
    loss_avg_train = utils.AverageMeter(device, args.epochs)

    #Val stats
    acc_val = utils.AverageMeter(device, args.epochs)
    loss_cls_avg_val = utils.AverageMeter(device, args.epochs)
    loss_uns_avg_val = utils.AverageMeter(device, args.epochs)
    loss_avg_val = utils.AverageMeter(device, args.epochs)

    # Training loop
    for epoch in tqdm(range(args.epochs)):
        net.train(True)
        #Train
        for i, data in enumerate(dataloader_train, 0):
            x = data[0].to(device)
            y = data [1].to(device)
            y.type(torch.float16)

            out, (z, dz_by_dx) = net(x)

            # y = nn.functional.one_hot(y, 10)
            loss_cls_train = criterion_cls(out, y)
            loss_uns_train = criterion_uns(device, z, dz_by_dx)
            loss_train = loss_cls_train + beta * loss_uns_train

            optimizer.zero_grad(set_to_none=True)
            loss_train.backward()
            optimizer.step()

        accuracy_train = utils.accuracy(out, y)
        acc_train.append(accuracy_train)
        loss_cls_avg_train.append(loss_cls_train)
        loss_uns_avg_train.append(loss_uns_train)
        loss_avg_train.append(loss_train)

        net.train(False)
        #Validation
        for i, data in enumerate(dataloader_valid, 0):
            x = data[0].to(device)
            # x = torch.nn.functional.normalize(x)
            y = data[1].to(device)
            y.type(torch.float16)

            out, (z, dz_by_dx) = net(x)

            loss_cls_val = criterion_cls(out, y)
            loss_uns_val = criterion_uns(device, z, dz_by_dx)
            loss_val = loss_cls_val + beta*loss_uns_val

        accuracy_val = utils.accuracy(out,y)
        acc_val.append(accuracy_val)
        loss_cls_avg_val.append(loss_cls_val)
        loss_uns_avg_val.append(loss_uns_val)
        loss_avg_val.append(loss_val)

        if epoch % args.display_freq == 0 :
            tqdm.write('Classification loss Train: {} Valid {}'.format(loss_cls_train, loss_cls_val))
            tqdm.write('Unsupervised loss Train: {} Valid {}'.format(loss_uns_train, loss_uns_val))
            tqdm.write('Total loss Train: {} Valid {}'.format(loss_train, loss_val))
            tqdm.write('Accuracy Train: {} Valid {}'.format(accuracy_train, accuracy_val))

        if accuracy_val > acc_best:
                acc_best = accuracy_val
                torch.save({'model_state_dict': net.state_dict()}, 'results.pt')

    return acc_train, loss_cls_avg_train, loss_uns_avg_train, loss_avg_train, acc_val, loss_cls_avg_val, loss_uns_avg_val, loss_avg_val


def test(device, net, dataloader):
    accuracy = 0
    for i, data in enumerate(dataloader, 0):
        x = data[0].to(device)
        # x = torch.nn.functional.normalize(x)
        y = data[1].to(device)
        y.type(torch.float16)

        out, _ = net(x)
        accuracy += utils.accuracy(out, y)
    accuracy = accuracy / len(dataloader)
    return accuracy


def adapt(device, net, dataloader, args):
    if args.test_opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_test)
    elif args.test_opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr_test)

    #Loss function
    criterion_uns = utils.det_likelihood

    acc = utils.AverageMeter(device, args.test_epochs)
    loss_avg = utils.AverageMeter(device, args.test_epochs)
    loss_best = 10e7
    for epoch in tqdm(range(args.test_epochs)):
        net.train()
        for i, data in enumerate(dataloader, 0):
            x = data[0].to(device)
            y = data[1].to(device)
            y.type(torch.float16)

            out, (z, dz_by_dx) = net(x)

            loss = criterion_uns(device, z, dz_by_dx)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        net.eval()
        accuracy = utils.accuracy(out, y)
        acc.append(accuracy)
        loss_avg.append(loss)

        if epoch % args.display_freq == 0 :
            tqdm.write('Unsupervised loss: {}'.format(loss_avg.avg))
            tqdm.write('Accuracy: {}'.format(acc.avg))

        if loss > loss_best:
                loss_best = loss
                torch.save({'model_state_dict': net.state_dict()}, 'results_test_time.pt')

    return acc, loss_avg