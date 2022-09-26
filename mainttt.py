from __future__ import print_function
import warnings
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import configuration
from utils.prepare_dataset import *
from utils.misc import *
from utils.metrics import *
from utils.test_helpers2 import *
from utils.prepare_dataset import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

def experiment(args):
    '-----------Loading the model--------------------------------'
    if args.train_setting == 'pre+flow':
        checkpoint = torch.load('/home/vhakim/projects/rrg-ebrahimi/vhakim/TTTCifar/Results/ckpt.pth')
        net, ext, head, ssh = build_model(args, checkpoint['model'])
    else:
        args.use_scheduler = True
        net, ext, head, ssh = build_model(args)

    '-----------Loading the data--------------------------------'
    _, teloader = prepare_test_data(args)
    _, trloader = prepare_train_data(args)

    '-----------Optimizer--------------------------------'
    if args.train_setting == 'classification':
        parameters = list(net.parameters())
        optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.train_setting == 'pre+flow':
        parameters = list(head.parameters())
        #optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizer = optim.Adam(parameters, lr=args.lr)

    if args.use_scheduler:
        print('Using scheduler')
        if args.train_setting == 'classification':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(150, 250), gamma=0.1)
        elif args.train_setting == 'pre+flow':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepoch)

    '-----------Loss Functions--------------------------------'
    criterion = FocalLoss(gamma=3, size_average=True)
    target = Normal(torch.tensor(0).float().cuda(), torch.tensor(1).float().cuda())
    criterion_uns = neg_log_likelihood_2d
    get_message(args.train_setting)

    '-----------Training--------------------------------'
    errors_cls = []
    ece_list = []
    loss_cls = []
    loss_uns = []
    train_loss = []
    print('Running...')
    if args.train_setting == 'classification':
        print('\t\tError (%)\t\tfocal loss\t\tECE')
    else:
        print('\t\t\tValidation unsupervised\t\tTraining unsupervised')

    for epoch in range(1, args.nepoch+1):
        if args.train_setting == 'classification':
            net.train()
            ssh.eval()
        else:
            net.eval()
            ssh.train()
            #head.train()
            #ssh.train()
        container = []
        for batch_idx, (inputs, labels) in enumerate(trloader):
            inputs_cls, labels_cls = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if args.train_setting == 'pre+flow':
                # Adding transformations
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
            losses_uns = test_with_uns(device, teloader, ssh, head, target)
            loss_uns.append(losses_uns)
            print(('Epoch %d/%d:' % (epoch, args.nepoch)).ljust(24) +
                  '%.2f\t\t%.2f' % (losses_uns, train_loss[-1]))
        else:
            err_cls, _, losses_cls, ece= test(device, criterion, teloader, net)
            ece_list.append(ece)
            errors_cls.append(err_cls)
            loss_cls.append(losses_cls)
            print(('Epoch %d/%d:' % (epoch, args.nepoch)).ljust(24) +
                  '%.2f\t\t%.2f\t\t%.2f' % (err_cls * 100, losses_cls, ece))

    state = {'err_cls': errors_cls, 'loss_uns': loss_uns, 'loss_cls':loss_cls, 'train_loss_uns':train_loss,
             'ece': ece_list, 'net': net.state_dict(), 'head': head.state_dict(),
             'optimizer': optimizer.state_dict()}
    if args.train_setting == 'pre+flow':
        if args.use_patches:
            torch.save(state, 'Results/flow_patches_1D_' + args.shared + '.pth')
        else:
            torch.save(state, '/home/vhakim/projects/rrg-ebrahimi/vhakim/TTTCifar/Results/cont_' + args.shared + '_' + str(args.lr) + '_cifar10.pth')
    else:
        torch.save(state, 'Results/pretraining.pth')


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    args = configuration.argparser()
    my_makedir(args.outf)

    experiment(args)

