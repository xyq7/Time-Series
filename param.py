import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', default=6, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--output_dim', default=2, type=int)
    parser.add_argument('--T',  default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    return parser
