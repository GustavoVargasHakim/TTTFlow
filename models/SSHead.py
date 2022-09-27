from torch import nn
import torch.nn.functional as F
import math
import copy

class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

class ExtractorHead(nn.Module):
	def __init__(self, ext, head):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head

	def forward(self, x, features=False):
		x = self.ext(x)
		if features:
			return self.head(x), x
		else:
			return self.head(x)

def extractor_from_layer1(net):
	layers = [net.conv1,  net.bn1, nn.ReLU(inplace=True), net.layer1]
	return nn.Sequential(*layers)

def extractor_from_layer3(net):
	layers = [net.conv1, net.bn1, nn.ReLU(inplace=True), net.layer1, net.layer2, net.layer3]#, net.bn, net.relu, net.avgpool]
	return nn.Sequential(*layers)

def extractor_from_layer2(net):
	layers = [net.conv1,  net.bn1, nn.ReLU(inplace=True), net.layer1, net.layer2]
	return nn.Sequential(*layers)

def extractor_from_layer4(net):
	layers = [net.conv1, net.bn1, nn.ReLU(inplace=True), net.layer1, net.layer2, net.layer3, net.layer4]
	return nn.Sequential(*layers)

def head_on_layer2(flow):
	from models.realnvp import RealNVP
	head = []
	if flow == 'realnvp':
		head.append(RealNVP(256, 256, 512, 2, True, 16, 'checkerboard'))
	return nn.Sequential(*head)

