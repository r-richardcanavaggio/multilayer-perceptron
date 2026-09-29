import numpy as np
import pandas


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    print(x[0])
    exponents = np.exp(x)
    print(exponents[0])

    sum_of_exponents = np.sum(exponents, axis=1)
    print(sum_of_exponents)
    probabilities = np.exp(x) / sum_of_exponents
    return probabilities

def main():
    df = pandas.read_csv('training_data.csv')

    first_col = df.iloc[:, 0]
    df = df.drop(columns=df.columns[0])

    X = df.to_numpy()

    """(Number of Inputs, Number of Perceptrons)"""
    W1 = np.zeros((30, 15))
    # print(W1)
    B1 = np.zeros(15)

    W2 = np.zeros((15, 15))
    B2 = np.zeros(15)

    W3 = np.zeros((15, 2))
    B3 = np.zeros(2)

    Z1 = sigmoid(np.dot(X, W1) + B1)
    Z2 = sigmoid(np.dot(Z1, W2) + B2)
    # print(Z2)
    Z3 = softmax(np.dot(Z2, W3) + B3)
    print(Z3)


if __name__ == "__main__":
    main()