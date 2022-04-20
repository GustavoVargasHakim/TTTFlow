import argparse

def argparser():
    parser = argparse.ArgumentParser()

    #Settings
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    #Dataset
    parser.add_argument('--n', type=int, default=500, help='Number of samples')
    parser.add_argument('--rot', type=int, default=50, help='Rotation of the moons')
    parser.add_argument('--tran', type=float, default=1.0, help='Translation of the moons')
    parser.add_argument('--sep', type=int, default=0.5, help='Separation between moons')

    #Training
    parser.add_argument('--lr-train', type=float, default=2e-2, help='Training learning rate')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--train-opt', type=str, default='Adam', help='Training optimizer')
    parser.add_argument('--display-freq', type=int, default=5, help='Performance display frequency (in epochs)')
    parser.add_argument('--beta', type=float, default=0.1, help='Unsupervised loss weight')

    #Test time adaptation
    parser.add_argument('--lr-test', type=float, default=1e-3, help='Test time learning rate')
    parser.add_argument('--test-epochs', type=int, default=1000, help='Test time adaptation epochs')
    parser.add_argument('--test-opt', type=str, default='Adam', help='Test time optimizer', choices=('Adam', 'sgd'))

    #Architecture
    parser.add_argument('--batch-norm', type=bool, default=False, help='Use batch norm after linear layers')
    parser.add_argument('--flow', type=str, default='glow', help='Type of flow to use', choices=('glow', 'nvp'))

    return parser.parse_args()
