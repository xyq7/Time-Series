import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='pair', choices=('single', 'pair', 'macro'))
    parser.add_argument('--model', default='lstm', choices=('arima', 'lstm'))

    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)

    parser.add_argument('--T',  default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=50, type=int)

    parser.add_argument('--save-epochs', default=10, type=int)
    parser.add_argument('--dir', default='./data/processed/', type=str)
    return parser


