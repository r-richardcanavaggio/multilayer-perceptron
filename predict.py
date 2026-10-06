from src.NeuralNetwork import NeuralNetwork
from src.Math import binary_cross_entropy
import numpy as np
import pandas


def main():
    data = pandas.read_csv('data/testing_data.csv').to_numpy()
    model = NeuralNetwork.from_npz('data/model_weights.npz')

    n = data.shape[0]
    y = data[:, 0]
    X = np.ascontiguousarray(data[:, 1:])

    raw_prediction = model.forward(X)
    loss = binary_cross_entropy(y, raw_prediction)

    print(f"Binary cross-entropy loss: {loss}")

    raw_prediction[:, [0, 1]] = raw_prediction[:, [1, 0]]
    prediction = np.argmax(raw_prediction, axis=1)

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for y_true, y_pred in zip(y, prediction):
        if y_true == 1 and y_pred == 1:
            true_positive += 1
        elif y_true == 0 and y_pred == 1:
            false_positive += 1
        elif y_true == 1 and y_pred == 0:
            false_negative += 1
        elif y_true == 0 and y_pred == 0:
            true_negative += 1
    error_matrix = np.array([[true_positive, false_negative], [false_positive, true_negative]])

    accuracy = (true_positive + true_negative) / n
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"Accuracy: {accuracy * 100:.2f}%\n"
          f"Precision: {precision * 100:.2f}%\n"
          f"Recall: {recall * 100:.2f}%\n"
          f"F1 Score: {f1_score * 100:.2f}%")

    


if __name__ == "__main__":
    main()