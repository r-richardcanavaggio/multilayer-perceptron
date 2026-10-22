import sys
import numpy as np
import pandas
from src.NeuralNetwork import NeuralNetwork
from src.Math import binary_cross_entropy
from src.Dataset import Dataset


def main() -> None:
    if len(sys.argv) > 3:
        print("Error: use as python3 "
              "predict.py [dataset.csv] [model_weights.npz]", file=sys.stderr)
        sys.exit(1)

    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/testing_data.csv'
    weights_path = (sys.argv[2] if len(sys.argv) > 2
                    else 'data/model_weights.npz')

    try:
        model = NeuralNetwork.from_npz(weights_path)
    except FileNotFoundError:
        print(f"Error: Model weights not found at '{weights_path}. "
              "Please run train.py first.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to load model from '{weights_path}: {e}",
              file=sys.stderr)
        sys.exit(1)

    test_dataset = Dataset(data_path)

    if model.scaler is not None:
        test_dataset.normalize(model.scaler)

    labels = test_dataset.labels

    raw_prediction = model.forward(test_dataset.parameters)
    loss = binary_cross_entropy(labels, raw_prediction)

    print(f"Binary cross-entropy loss: {loss}\n")

    prediction = np.argmax(raw_prediction, axis=1)

    true_positive = np.sum((labels == 1) & (prediction == 1))
    true_negative = np.sum((labels == 0) & (prediction == 0))
    false_positive = np.sum((labels == 0) & (prediction == 1))
    false_negative = np.sum((labels == 1) & (prediction == 0))

    error_matrix = pandas.DataFrame(
        [[true_positive, false_negative],
         [false_positive, true_negative]],
        index=['Actual Positives', 'Actual Negatives'],
        columns=['Predicted Positives', 'Predicted Negatives']
    )

    print(error_matrix)
    print()

    n = len(test_dataset)
    eps = 1e-15

    accuracy = (true_positive + true_negative) / n
    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive / (true_positive + false_negative + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
    }

    for name, score in metrics.items():
        print(f"{name:<10}: {score * 100:.2f}%")


if __name__ == "__main__":
    main()
