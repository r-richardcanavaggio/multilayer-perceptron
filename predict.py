from src.NeuralNetwork import NeuralNetwork
from src.Math import binary_cross_entropy
import numpy as np
import pandas


def main():
    data = pandas.read_csv('data/testing_data.csv').to_numpy()
    model = NeuralNetwork.from_npz('model_weights.npz')

    y = data[:, 0]
    X = np.ascontiguousarray(data[:, 1:])

    forward_pass = model.forward(X)
    loss = binary_cross_entropy(y, forward_pass)
    print(loss)

    


if __name__ == "__main__":
    main()