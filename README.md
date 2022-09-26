# TTTFlow

Official reposiroty of the WACV 2023 paper "TTTFlow: Unsupervised Test-Time Training with Normalizing Flows", by David Osowiechi, Gustavo A. Vargas Hakim, Mehrdad Noori, Milad Cheraghalikhani, Ismail Ben Ayed, and Christian Desrosiers.
This work was greatly inspired by the code in [TTT](https://github.com/yueatsprograms/ttt_cifar_release).

TTTFlow uses a Normalizing Flow on top of source-trained feature extractor in order to learn a mapping from the source distribution to a Gaussian distribution. At test-time, the Normalizing Flow is used as a domain shift detector to modify the weights of the feature extractor according to the new domain, and perform classification. 

![Diagram](https://github.com/GustavoVargasHakim/TTTFlow/blob/master/TTTFlow.png)

Please follow the following instructions in order to reproduce the experiments.

## Setup 

### Requirements

This project was developed using Python 3.8.10, PyTorch 1.11.2 and CUDA 11.1. Please visit the requirements
file "requirements.txt", or use thw following command:

`pip install -r requirements.txt`

### Datasets

The experiments utilize the CIFAR-10 training split as the source dataset. It can be downloaded from 
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), or can also be done using torchvision
datasets: `train_data = torchvision.datasets.CIFAR10(root='Your/Path/To/Data', train=True, download=True)`.
The same line of code can be used to load the data if it is already downloaded, just by changing the
argument `download` to `False`.

At test-time, we use CIFAR-10-C and CIFAR-10-new. The first one can be downloaded from [CIFAR-10-C](
https://zenodo.org/record/2535967#.YzHFMXbMJPY). For the second one, please download the files 
`cifar10.1_v6_data.npy` and `cifar10.1_v6_labels.npy` from [CIFAR-10-new](https://github.com/modestyachts/CIFAR-10.1/tree/master/datasets).
All the data should be placed in a common folder from which they can be loaded, e.g., `/datasets/`.
