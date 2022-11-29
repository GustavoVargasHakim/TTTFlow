# TTTFlow

Official repository of the WACV 2023 paper "TTTFlow: Unsupervised Test-Time Training with Normalizing Flows", by David Osowiechi, Gustavo A. Vargas Hakim, Mehrdad Noori, Milad Cheraghalikhani, Ismail Ben Ayed, and Christian Desrosiers.
The whole article can be found [here](https://arxiv.org/abs/2210.11389).
This work was greatly inspired by the code in [TTT](https://github.com/yueatsprograms/ttt_cifar_release).

TTTFlow uses a Normalizing Flow on top of source-trained feature extractor in order to learn a mapping from the source distribution to a Gaussian distribution. At test-time, the Normalizing Flow is used as a domain shift detector to modify the weights of the feature extractor according to the new domain, and perform classification. 

![Diagram](https://github.com/GustavoVargasHakim/TTTFlow/blob/master/TTTFlow.png)

Please follow the following instructions in order to reproduce the experiments.

## Setup 

### Requirements

This project was developed using Python 3.8.10, PyTorch 1.11.2 and CUDA 11.1. Please visit the requirements
file "requirements.txt", or use the following command:

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

### File roots

There are certain files in which the specific paths to load/save files need to be specified. This files are in the root of the project: `main.py`, `configuration.py`, and inside the folder "test_calls": `configuration.py`, `test_adapt.py`, and `test_adapt_all.py`. You can see the dummy path `/path/to/project/` in the strings where you should change to your actual path.

## Source Training

The file `main.py` can be used for training the ResNet-50 model on CIFAR-10 classification, and also to train the Normalizing Flow on top of the feature extractor. To choose each option, please edit the `configuration.py` file, and for the parameter called `--train-setting`, choose between "classification" or "pre+flow", corresponding to classification training and flow training, respectively. For the Normalizing Flow training, there is also the option of using the contrastive pretraining weights from [TTT++](https://github.com/vita-epfl/ttt-plus-plus). This can be done by changing the value of the configuration parameter `--pretrain` to "contrastive", instead of "normal". Notice that for the contrastive weights, it is not necessary to train for classification. Make sure of changing the different hyperparameters according to the paper to reproduce the results. 

## Test-time Adaptation

At test-time, we utilize the files inside the folder "test_calls", with its own configuration file. You can use two files:

- `test_adapt.py`: to adapt to an individual corruption from CIFAR-10-C, or CIFAR-10-new. Change the parameter `--corruption` inside the `configuration.py` file to `cifar-new` or to any of the 15 corruption names in CIFAR-10-C (please see the file `prepare_dataset.py` inside folder "utils" for more information on the names).

- `test_adapt_all.py`: to adapt to all the corruptions from CIFAR-10-C in a row. It also evaluates the method for different number of iterations (3, 5, 10, 20), but you can change the values in line 71. The execution for this experiment takes considerably longer than only using `test_adapt.py`, and the execution time grows with the increasing number of iterations. 

## Citation

If you found this repository, or its related paper useful for your research, you can cite this work as:

```
@inproceedings{TTTFlow2023,
  title={TTTFlow: Unsupervised Test-Time Training with Normalizing Flow},
  author={David Osowiechi and Gustavo A. Vargas Hakim and Milad Cheraghalikhani and Mehrdad Noori and Ismail Ben Ayed and Christian Desrosiers},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={},
  month={January},
  year={2023}
}
```

