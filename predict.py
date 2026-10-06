from src.NeuralNetwork import NeuralNetwork
from src.Math import binary_cross_entropy
import numpy as np
import pandas


def main():
    data = pandas.read_csv('data/testing_data.csv').to_numpy()
    model = NeuralNetwork.from_npz('data/model_weights.npz')

    y = data[:, 0]
    X = np.ascontiguousarray(data[:, 1:])

    raw_prediction = model.forward(X)
    y_pred = np.argmax(raw_prediction, axis=1)

    print(y_pred)
    print(y)

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for x, y in zip(y_pred, y):
        

    


if __name__ == "__main__":
    main()