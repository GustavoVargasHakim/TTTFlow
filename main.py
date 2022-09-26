from __future__ import print_function
import sys
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.distributions import Normal
import numpy as np
sys.path.append('/home/vhakim/projects/rrg-ebrahimi/vhakim/TTTCifar/')

import configuration
from utils import misc
from utils import test_helpers, test_helpers_contrastive
from utils import prepare_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

def experiment(args):
    print(args.dataset)
    '-----------Loading the model--------------------------------'
    if args.train_setting == 'pre+flow':
        if args.pretrain == 'normal':
            checkpoint = torch.load('/path/to/weights/TTTFlow/Results/pretraining_CIFAR10.pth')
            net, ext, head, ssh = test_helpers.build_model(args, checkpoint['model'])
        else:
            checkpoint = torch.load('/path/to/weights/TTTFlow/Results/ckpt.pth')
            net, ext, head, ssh = test_helpers_contrastive.build_model(args, checkpoint['model'])
    else:
        args.use_scheduler = True
        net, ext, head, ssh = test_helpers.build_model(args)

    '-----------Loading the data--------------------------------'
    _, teloader = prepare_dataset.prepare_test_data(args)
    _, trloader = prepare_dataset.prepare_train_data(args)

    '-----------Optimizer--------------------------------'
    if args.train_setting == 'classification':
        parameters = list(net.parameters())
    elif args.train_setting == 'pre+flow':
        parameters = list(head.parameters())
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.use_scheduler:
        print('Using scheduler')
        if args.train_setting == 'classification':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(150, 250), gamma=0.1)
        elif args.train_setting == 'pre+flow':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepoch)

    '-----------Loss Functions--------------------------------'
    criterion = nn.CrossEntropyLoss()
    target = Normal(torch.tensor(0).float().cuda(), torch.tensor(1).float().cuda())
    criterion_uns = misc.neg_log_likelihood_2d
    misc.get_message(args.train_setting)

    '-----------Training--------------------------------'
    errors_cls = []
    loss_cls = []
    losses_uns = []
    train_loss = []
    print('Running...')
    if args.train_setting == 'classification':
        print('\t\tError (%)\t\tfocal loss\t\tECE')
    else:
        print('\t\t\tValidation Loss\t\tTraining Loss')

    for epoch in range(1, args.nepoch+1):
        if args.train_setting == 'classification':
            net.train()
            ssh.eval()
        else:
            ssh.train()
            net.eval()
        container = []
        for batch_idx, (inputs, labels) in enumerate(trloader):
            inputs_cls, labels_cls = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if args.train_setting == 'pre+flow':
                z, log_det = ssh(inputs_cls)
                loss = criterion_uns(target, z, log_det)

            else:
                outputs_cls = net(inputs_cls)
                loss = criterion(outputs_cls, labels_cls)

            container.append(loss.detach().cpu())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if args.use_scheduler:
            scheduler.step()

        train_loss.append(np.array(container).mean())

        '''Displaying and saving parameters'''
        if args.train_setting == 'pre+flow':
            loss_uns = test_helpers.test_with_uns(device, teloader, ssh, head, target)
            losses_uns.append(loss_uns)
            print(('Epoch %d/%d:' % (epoch, args.nepoch)).ljust(24) + '%.2f\t\t\t%.2f' % (loss_uns, train_loss[-1]))
        else:
            err_cls, _, losses_cls= test_helpers.test(device, criterion, teloader, net)
            errors_cls.append(err_cls)
            loss_cls.append(losses_cls)
            print(('Epoch %d/%d:' % (epoch, args.nepoch)).ljust(24) +
                  '%.2f\t\t%.2f' % (err_cls * 100, losses_cls))

        state = {'err_cls': errors_cls, 'loss_uns': losses_uns, 'loss_cls':loss_cls, 'train_loss_uns':train_loss,
            'model': net.state_dict(), 'head': head.state_dict(), 'ext': ext.state_dict(),
             'optimizer': optimizer.state_dict()}

        if args.train_setting == 'pre+flow':
            if args.pretrain == 'normal':
                torch.save(state, '/path/to/project/TTTFlow/Results/flow_' + str(args.shared) + '_' + '_cifar10.pth')
            elif args.pretrain == 'contrastive':
                torch.save(state, '/path/to/project/TTTFlow/Results/flow_' + str(args.shared) + '_contrastive' + '_cifar10.pth')

        else:
            torch.save(state, '/path/to/project/TTTFlow/Results/pretraining_CIFAR10.pth')

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    args = configuration.argparser()
    misc.my_makedir(args.outf)

    experiment(args)

