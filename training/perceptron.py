import numpy as np
import pandas


def sigmoid( x: float ) -> float:
    return 1 / (1 + np.exp(-x))

def main():
    df = pandas.read_csv('training_data.csv')

    first_col = df.iloc[:, 0]
    df = df.drop(columns=df.columns[0])

    weights = np.array([0.05, 0.00, -0.47])
    bias: float = 0.25
    inputs = np.array([1.0, 2.0, 3.0])

    print(weights * inputs)
    output = sigmoid(np.sum(weights * inputs) + bias)
    print(output)




if __name__ == "__main__":
    main()