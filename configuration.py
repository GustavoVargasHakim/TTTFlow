import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--dataroot', default='/your/root/to/data')
    parser.add_argument('--corruption', default='original')
    parser.add_argument('--shared', default='layer2')
    parser.add_argument('--pretrain', default='normal', choices=('normal', 'contrastive'))
    parser.add_argument('--train-setting', type=str, default='pre+flow', help='Training setting', choices=('pre+flow','classification'))
    ########################################################################
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    ########################################################################
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--nepoch', default=100, type=int)
    parser.add_argument('--use-scheduler', default=True, help='To use learning rate scheduling or not')
    ########################################################################
    parser.add_argument('--outf', default='.')

    return parser.parse_args()
