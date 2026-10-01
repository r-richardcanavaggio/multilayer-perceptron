import numpy as np
import pandas
import argparse
from DenseLayer import DenseLayer
from Math import binary_cross_entropy
from NeuralNetwork import NeuralNetwork


def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, nargs="+", help="Size of layers")
    parser.add_argument("--epochs", type=int, help="Number of epochs for training")
    parser.add_argument("--loss", type=str, help="Error function to use during training")
    parser.add_argument("--batch_size", type=int, help="Size of input data to use")
    parser.add_argument("--learning_rate", type=float, help="Floating point value of learning rate")

    args = parser.parse_args()

    df = pandas.read_csv('training_data.csv')

    # y_true = df.iloc[:, 0].to_numpy()
    # y_true_inv = 1 - y_true
    # y_true_double = np.stack((y_true, y_true_inv), axis=1)

    # df = df.drop(columns=df.columns[0])

    X = df.to_numpy()

    mini_batch = args.batch_size

    couches = [
        DenseLayer(30, 15, mini_batch, activation='sigmoid', weights_initializer='glorot'),
        DenseLayer(15, 15, mini_batch, activation='sigmoid', weights_initializer='glorot'),
        DenseLayer(15, 2, mini_batch, activation='softmax', weights_initializer='glorot')
    ]

    model = NeuralNetwork(couches, learning_rate=args.learning_rate)

    for epoch in range(args.epochs):
        for first in range(0, len(X), mini_batch):
            subset = X[first:first+mini_batch]
            
            y_true = subset[:, 0]
            y_true_inv = 1 - y_true
            y_true_double = np.stack((y_true, y_true_inv), axis=1)

            subset = np.delete(subset, 0, axis=1)

            forward_pass = model.forward(subset)

            loss = binary_cross_entropy(y_true, forward_pass)
            print(f"Epoch {epoch}: Loss = {loss}")

            d_Z = forward_pass - y_true_double
            backward_pass = model.backward(d_Z)


if __name__ == "__main__":
    main()