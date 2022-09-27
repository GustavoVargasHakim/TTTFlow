import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--level', default=5, type=int, help='Severity level in corruption')
    parser.add_argument('--corruption', default='cifar_new')
    parser.add_argument('--dataroot', default='/path/to/dataset/datasets')
    parser.add_argument('--shared', default='layer2')
    ########################################################################
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--adapt', type=bool, default=True)
    ########################################################################
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--niter', default=10, type=int)
    parser.add_argument('--online', type=bool, default=False)
    parser.add_argument('--threshold', default=10, type=float)
    ########################################################################
    parser.add_argument('--outf', default='.')

    return parser.parse_args()
