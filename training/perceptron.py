import numpy as np
import pandas
import argparse
import matplotlib.pyplot as plt
from DenseLayer import DenseLayer
from Math import binary_cross_entropy
from NeuralNetwork import NeuralNetwork


def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, nargs="+", help="Size of layers")
    parser.add_argument("--epochs", type=int, help="Number of epochs for training")
    parser.add_argument("--loss", type=str, default='sigmoid', help="Error function to use during training")
    parser.add_argument("--batch_size", type=int, help="Size of input data to use")
    parser.add_argument("--learning_rate", type=float, help="Floating point value of learning rate")
    parser.add_argument("--weights_initializer", type=str, default='glorot', help="Algorithm of weights initialisation. Defaults to Xavier/Glorot Init")

    args = parser.parse_args()

    df = pandas.read_csv('training_data.csv').to_numpy()

    y = df[:, 0]
    X = np.ascontiguousarray(df[:, 1:])
    y_double = np.column_stack((y, 1 - y))

    mini_batch = args.batch_size
    input_size = X.shape[1]

    couches = []
    for l in args.layer:
        current_layer = DenseLayer(input_size, l, mini_batch, activation=args.loss, weights_initializer='glorot')
        couches.append(current_layer)
        input_size = l
    couches.append(DenseLayer(args.layer[-1], 2, mini_batch, activation='softmax', weights_initializer='glorot'))

    model = NeuralNetwork(couches, learning_rate=args.learning_rate)

    losses = []

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
        losses.append(total_loss)
        print(f"Epoch {epoch}: Loss = {total_loss}")

    x = [i for i in range(args.epochs)]
    plt.plot(x, losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.show()



if __name__ == "__main__":
    main()