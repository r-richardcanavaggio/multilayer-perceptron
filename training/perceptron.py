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

    df = pandas.read_csv('training_data.csv').to_numpy()

    y = df[:, 0]
    X = np.ascontiguousarray(df[:, 1:])
    y_double = np.column_stack((y, 1 - y))

    mini_batch = args.batch_size

    couches = [
        DenseLayer(30, 15, mini_batch, activation='sigmoid', weights_initializer='glorot'),
        DenseLayer(15, 15, mini_batch, activation='sigmoid', weights_initializer='glorot'),
        DenseLayer(15, 2, mini_batch, activation='softmax', weights_initializer='glorot')
    ]

    model = NeuralNetwork(couches, learning_rate=args.learning_rate)

    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0

        for first in range(0, len(X), mini_batch):
            X_batch = X[first : first + mini_batch]
            y_batch = y[first : first + mini_batch]
            y_double_batch = y_double[first : first + mini_batch]

            forward_pass = model.forward(X_batch)

            loss = binary_cross_entropy(y_batch, forward_pass)
            total_loss += loss
            num_batches += 1

            d_Z = forward_pass - y_double_batch
            backward_pass = model.backward(d_Z)

        total_loss /= num_batches
        print(f"Epoch {epoch}: Loss = {total_loss}")
        
        


if __name__ == "__main__":
    main()