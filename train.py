import numpy as np
import pandas
import argparse
import matplotlib.pyplot as plt
from src.DenseLayer import DenseLayer
from src.Math import binary_cross_entropy
from src.NeuralNetwork import NeuralNetwork


def init_parser():
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
        "-L", "--loss",
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

    return parser.parse_args()


def init_nn(input_size: int, layers_sizes: list[int],
            loss_function: str,
            learning_rate: float) -> NeuralNetwork:
    layers = []

    for layer in layers_sizes:
        new_layer = DenseLayer(
            input_size, layer, activation=loss_function,
            weights_initializer='glorot'
        )
        layers.append(new_layer)
        input_size = layer
    layers.append(DenseLayer(
        layers_sizes[-1], 2, activation='softmax',
        weights_initializer='glorot'
    ))

    return NeuralNetwork(layers, learning_rate=learning_rate)


def loss_plot(epochs: int, losses_train: list[float], losses_val: list[float]):
    x = [i for i in range(epochs)]
    plt.plot(x, losses_train, color='blue', label='training loss')
    plt.plot(x, losses_val,
             color='orange', linestyle='--',
             label='validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def accuracy_plot(epochs: int, accuracy_train: list[float], accuracy_val: list[float]):
    x = [i for i in range(epochs)]
    plt.plot(x, accuracy_train, color='blue', label='training loss')
    plt.plot(x, accuracy_val,
                color='orange', linestyle='--',
                label='validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raw_predict = y_pred.copy()
    raw_predict[:, [0, 1]] = raw_predict[:, [1, 0]]
    prediction = np.argmax(raw_predict, axis=1)
    return (y_true == prediction).sum() / y_pred.shape[0]


def main():
    np.random.seed(42)

    args = init_parser()

    training_data = pandas.read_csv('data/training_data.csv').to_numpy()
    validation_data = pandas.read_csv('data/testing_data.csv').to_numpy()

    # First column of training_data, real output
    y_train = training_data[:, 0]
    # Output in a binary format, M = [1, 0] B = [0, 1]
    y_train_double = np.column_stack((y_train, 1 - y_train))
    y_val = validation_data[:, 0]

    X = np.ascontiguousarray(training_data[:, 1:])
    X_val = np.ascontiguousarray(validation_data[:, 1:])

    mini_batch = args.batch_size
    input_size = X.shape[1]
    width = len(str(args.epochs))

    model = init_nn(input_size, args.layers,
                    args.loss, args.learning_rate)

    losses_train = []
    losses_val = []
    accuracies_train = []
    accuracies_val = []

    for epoch in range(args.epochs):
        total_loss_train = 0
        total_accuracy_train = 0.
        num_batches = 0

        for first in range(0, len(X), mini_batch):
            X_batch = X[first: first + mini_batch]
            y_batch = y_train[first: first + mini_batch]
            y_double_batch = y_train_double[first: first + mini_batch]

            forward_pass = model.forward(X_batch)

            loss = binary_cross_entropy(y_batch, forward_pass)
            total_loss_train += loss

            accuracy = compute_accuracy(y_batch, forward_pass)
            total_accuracy_train += accuracy

            num_batches += 1

            d_Z = forward_pass - y_double_batch
            model.backward(d_Z)

        total_loss_train /= num_batches
        total_accuracy_train /= num_batches
        losses_train.append(total_loss_train)
        accuracies_train.append(total_accuracy_train)

        pred_validation = model.forward(X_val)

        loss_val = binary_cross_entropy(y_val, pred_validation)
        losses_val.append(loss_val)

        accuracy_val = compute_accuracy(y_val, pred_validation)
        accuracies_val.append(accuracy_val)

        print(f"Epoch {epoch + 1:>{width}}: "
              f"Train loss = {total_loss_train:.6f} | "
              f"Validation loss = {loss_val:.6f}")

    loss_plot(args.epochs, losses_train, losses_val)
    accuracy_plot(args.epochs, accuracies_train, accuracies_val)
    model.export()


if __name__ == "__main__":
    main()
