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
    parser.add_argument(
        "-s", "--stop_early",
        nargs='?', type=int, const=10, default=False,
        help="Training will stop after N number of epochs"
        "where validation stopped diminishing."
        "Inactive by default. If flag is present,"
        "defaults to 10."
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9,
        help="Beta1 for Adam"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999,
        help="Beta2 for Adam"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8,
        help="Epsilon value for Adam"
    )

    return parser.parse_args()
