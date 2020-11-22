import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='single', choices=('single', 'pair', 'macro'))
    parser.add_argument('--model', default='lstm', choices=('arima', 'lstm'))

    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)

    parser.add_argument('--T',  default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)

    parser.add_argument('--dir', default='./data/processed/', type=str)
    return parser
