import sys
import numpy as np
import pandas
from src.NeuralNetwork import NeuralNetwork
from src.Math import binary_cross_entropy


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

    try:
        raw_df = pandas.read_csv(data_path)
        if raw_df.empty:
            print(f"Error: Dataset '{data_path}' is empty.", file=sys.stderr)
            sys.exit(1)
        data = raw_df.to_numpy()
    except FileNotFoundError:
        print(f"Error: Dataset not found at '{data_path}'. "
              "Please run split.py first.", file=sys.stderr)
        sys.exit(1)
    except pandas.errors.EmptyDataError:
        print(f"Error: Dataset '{data_path}' is empty.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read dataset from '{data_path}': {e}",
              file=sys.stderr)
        sys.exit(1)

    n = data.shape[0]
    if n == 0 or data.shape[1] < 2:
        print(f"Error: '{data_path}' does not contain enough data or columns.",
              file=sys.stderr)
        sys.exit(1)

    labels = data[:, 0]
    input = np.ascontiguousarray(data[:, 1:])

    expected_features = model.layers[0].weights.shape[0]
    if input.shape[1] != expected_features:
        print(f"Error: Feature mismatch."
              f"Model expects {expected_features} features, "
              f"got {input.shape[1]} from '{data_path}'.", file=sys.stderr)
        sys.exit(1)

    raw_prediction = model.forward(input)
    loss = binary_cross_entropy(labels, raw_prediction)

    print(f"Binary cross-entropy loss: {loss}\n")

    prediction = np.argmax(raw_prediction, axis=1)

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for y_true, y_pred in zip(labels, prediction):
        if y_true == 1 and y_pred == 1:
            true_positive += 1
        elif y_true == 0 and y_pred == 1:
            false_positive += 1
        elif y_true == 1 and y_pred == 0:
            false_negative += 1
        elif y_true == 0 and y_pred == 0:
            true_negative += 1

    row_names = ['Actual Positives', 'Actual Negatives']
    col_names = ['Predicted Positives', 'Predicted Negatives']
    error_matrix = np.array([[true_positive, false_negative],
                             [false_positive, true_negative]])
    error_df = pandas.DataFrame(error_matrix,
                                index=row_names, columns=col_names)

    print(error_df)
    print()

    accuracy = (true_positive + true_negative) / n if n > 0 else 0.0
    precision = (true_positive / (true_positive + false_positive)
                 if (true_positive + false_positive) > 0 else 0.0)
    recall = (true_positive / (true_positive + false_negative)
              if (true_positive + false_negative) > 0 else 0.0)
    f1_score = (2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0 else 0.0)

    print(f"Accuracy: {accuracy * 100:.2f}%\n"
          f"Precision: {precision * 100:.2f}%\n"
          f"Recall: {recall * 100:.2f}%\n"
          f"F1 Score: {f1_score * 100:.2f}%")


if __name__ == "__main__":
    main()
