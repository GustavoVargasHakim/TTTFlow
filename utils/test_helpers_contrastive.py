import numpy as np

from utils.misc import *
from models.SSHead import extractor_from_layer1, extractor_from_layer2, extractor_from_layer3, extractor_from_layer4
from models.realnvp import RealNVP
from models.BigResNet import SupConResNet, LinearClassifier
from models.SSHead import ExtractorHead


def build_resnet50(args):
    print('Building ResNet50...')
    if args.dataset == 'cifar10':
        classes = 10
    classifier = LinearClassifier(num_classes=classes).cuda()
    ssh = SupConResNet().cuda()
    head = ssh.head
    ext = ssh.encoder
    net = ExtractorHead(ext, classifier).cuda()

    return net, ext, head, ssh, classifier

def build_model(args, state_dict=None):
    print('Building model...')
    if args.dataset == 'cifar10':
        net, _, _, _, _ = build_resnet50(args)
    if state_dict is not None:
        print('Using pre-trained classifier with contrastive learning')
        net_dict = {}
        for k, v in state_dict.items():
            if k[:4] != 'head':
                k = k.replace("encoder.", "ext.")
                k = k.replace("fc.", "head.fc.")
                net_dict[k] = v
        net.load_state_dict(net_dict)

    if args.shared == 'layer1':
        ext = extractor_from_layer1(net.ext)
        head = RealNVP(256, 256, 512, 2, True, 32, 'checkerboard')

    elif args.shared == 'layer2':
        ext = extractor_from_layer2(net.ext)
        head = nn.Sequential(RealNVP(512, 512, 1024, 2, True, 16, 'checkerboard'))

    elif args.shared == 'layer3':
        ext = extractor_from_layer3(net.ext)
        head = RealNVP(1024, 1024, 2048, 2, True, 8, 'checkerboard')

    elif args.shared == 'layer4':
        ext = extractor_from_layer4(net.ext)
        head = RealNVP(2048, 2048, 4096, 2, True, 4, 'checkerboard')

    ssh = ExtractorHead(ext, head).cuda()

    return net, ext, head, ssh


def test(device, criterion, dataloader, net):
        net.eval()
        correct = []
        losses_cls = []
        for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.no_grad():
                        pred = net(inputs) #Prediction
                        loss_cls = criterion(pred, labels) #Loss function
                        losses_cls.append(loss_cls.cpu())
                        _, predicted = pred.max(dim=1)
                        correct.append(predicted.eq(labels).cpu())#Correctness
        correct = torch.cat(correct).numpy()
        losses_cls = np.array(losses_cls)
        net.train()
        return 1-correct.mean(), correct, losses_cls.mean()

def test_with_uns(device, dataloader, ssh, head, target):
        criterion_uns = neg_log_likelihood_2d
        head.eval()
        losses_uns = []
        for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.no_grad():
                        z, log_det = ssh(inputs)
                        loss_uns = criterion_uns(target, z, log_det)
                        losses_uns.append(loss_uns.cpu())
        losses_uns = torch.tensor(losses_uns).numpy()
        head.train()
        return losses_uns.mean()
