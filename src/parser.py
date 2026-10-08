import argparse


def init_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l", "--layers",
        type=int, default=[24, 24, 24],
        nargs="+", help="Size of layers"
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int, default=84,
        help="Number of epochs for training"
    )
    parser.add_argument(
        "-a", "--activation",
        type=str, default='sigmoid',
        help="Error function to use during training"
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int, default=8,
        help="Size of input data to use"
    )
    parser.add_argument(
        "-lR", "--learning_rate",
        type=float, default=0.314,
        help="Floating point value of learning rate"
    )
    parser.add_argument(
        "-w", "--weights_initializer",
        type=str, default='glorot',
        help="Weights initialisation. Defaults to Xavier/Glorot Init"
    )
    parser.add_argument(
        "-o", "--optimizer",
        type=str, default='SGD', choices=['SGD', 'Adam'],
        help="Optimizer algorithm during backpropagation."
        "Defaults to Stochastic Gradient Descent"
    )

    return parser.parse_args()
