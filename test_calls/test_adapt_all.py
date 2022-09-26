from __future__ import print_function
from tqdm import tqdm
import torch.optim as optim
from torch.distributions import Normal
import torch.backends.cudnn as cudnn
import sys
sys.path.append('/path/to/project/TTTFlow/')

import configuration
from utils.misc import *
from utils import test_helpers, test_helpers_contrastive
from utils import prepare_dataset

args = configuration.argparser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
my_makedir(args.outf)

'''--------------------Loading Model-----------------------------'''
print('Test-time adaptation')
if args.pretrain == 'normal':
	checkpoint = torch.load('/path/to/project/TTTFlow/Results/pretraining_CIFAR10.pth')
	ckpt = torch.load('/path/to/project/TTTFlow/Results/flow_layer2_cifar10.pth')
	net, ext, head, ssh = test_helpers.build_model(args, checkpoint['model'])
elif args.pretrain == 'contrastive':
	checkpoint = torch.load('/path/to/project/TTTFlow/Results/ckpt.pth')
	ckpt = torch.load('/path/to/project/TTTFlow/Results/flow_layer2_contrastive_cifar10.pth')
	net, ext, head, ssh = test_helpers_contrastive.build_model(args, checkpoint['model'])
head.load_state_dict(ckpt['head'])

'''--------------------Loss Function-----------------------------'''
target = Normal(torch.tensor(0).float().cuda(), torch.tensor(1).float().cuda())
criterion_uns = neg_log_likelihood_2d
optimizer_uns = optim.SGD(ext.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

'''--------------------Adaptation Function-----------------------------'''
def adapt_batch(device, ext, ssh, target, inputs, args):
	ssh.eval()
	ext.train()
	for iteration in range(args.niter):
		inputs = inputs.to(device, non_blocking=True)
		optimizer_uns.zero_grad(set_to_none=True)
		z, log_det = ssh(inputs)
		loss = criterion_uns(target, z, log_det)
		loss.backward()
		optimizer_uns.step()

'''--------------------Testing Function-----------------------------'''
def test_batch(device, net, ssh, image, labels, target):
	net.eval()
	ssh.eval()
	inputs, labels = image.to(device, non_blocking=True), labels.to(device, non_blocking=True)
	with torch.no_grad():
		outputs = net(inputs)
		z, log_det = ssh(inputs)
		predicted = torch.argmax(outputs, dim=1)
		softmax = torch.softmax(outputs, dim=1).cpu().detach().numpy()
		loss_uns = criterion_uns(target, z, log_det)
		correctness = predicted.eq(labels).cpu()

	return correctness, loss_uns.cpu().detach().item(), softmax

'''--------------------Test-Time Adaptation-----------------------------'''
print('Running...')
adaptations = 0
good_good = []
good_bad = []
bad_good = []
bad_bad = []
correct = 0
iterations = [1,3,10,20]
corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
               'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

for iter in iterations:
    args.niter = iter
    for corr in corruptions:
        args.corruption = corr
        _, teloader = prepare_dataset.prepare_test_data(args)
        for batch_idx, (inputs, labels) in tqdm(enumerate(teloader)):
            if not args.online:
                net.load_state_dict(ckpt['net'])
                head.load_state_dict(ckpt['head'])

            correctness, loss_uns, softmax = test_batch(device, net, ssh, inputs, labels, target)

            if args.adapt:
                adaptations += 1
                adapt_batch(device, ext, ssh, target, inputs, args)
                correctness_new, loss_uns, softmax = test_batch(device, net, ssh, inputs, labels, target)
                for i in range(len(correctness_new.tolist())):
                    if correctness[i] == True and correctness_new[i] == True:
                        good_good.append(1)
                    elif correctness[i] == True and correctness_new[i] == False:
                        good_bad.append(1)
                    elif correctness[i] == False and correctness_new[i] == True:
                        bad_good.append(1)
                    elif correctness[i] == False and correctness_new[i] == False:
                        bad_bad.append(1)
            else:
                correct += correctness.sum().item()

        correct += len(good_good) + len(bad_good)
        accuracy = correct/len(teset)

        print('--------------------RESULTS----------------------')
        print('Perturbation: ', args.corruption)
        if args.adapt:
            print('Total adaptations: ', adaptations)
            print('No. of epochs: ', args.niter)
            print('Good first, good after: ', len(good_good))
            print('Good first, bad after: ', len(good_bad))
            print('Bad first, good after: ', len(bad_good))
            print('Bad first, bad after: ', len(bad_bad))
        print('Accuracy: ', accuracy)
        print('Error: ', 1 - accuracy)
