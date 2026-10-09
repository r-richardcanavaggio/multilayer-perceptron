import numpy as np
import pandas
import sys
from src.parser import init_parser
from src.plot import accuracy_plot, loss_plot
from src.NeuralNetwork import NeuralNetwork


def ft_load(path: str) -> pandas.DataFrame:
    try:
        df = pandas.read_csv(path)
        print(f"Successfully loaded data in {path}")
        return df
    except FileNotFoundError:
        print(f"FileNotFoundError: {path}")
        sys.exit()
    except PermissionError:
        print(f"PermissionError: denied {path}")
        sys.exit()
    except pandas.errors.ParserError:
        print(f"Error: Data Parsing Error. Data might be corrupted in {path}")
        sys.exit()
    except UnicodeDecodeError:
        print(f"UnicodeDecodeError: File could not be decoded at {path}")
        sys.exit()
    except Exception as e:
        print(f"Unexepected error while reading file {e}")
        sys.exit()


def load_data(train_path: str, val_path: str):
    training_data = ft_load(train_path).to_numpy()
    validation_data = ft_load(val_path).to_numpy()

    # First column of training_data, real output
    y_train = training_data[:, 0]
    # Output in a binary format, M = [0, 1] B = [1, 0]
    y_train_double = np.column_stack((1 - y_train, y_train))
    y_val = validation_data[:, 0]

    X_train = np.ascontiguousarray(training_data[:, 1:])
    X_val = np.ascontiguousarray(validation_data[:, 1:])

    return X_train, y_train, y_train_double, X_val, y_val


def main():
    np.random.seed(42)

    args = init_parser()
    X, y, y_double, X_val, y_val = load_data(
        'data/training_data.csv', 'data/testing_data.csv'
        )

    model = NeuralNetwork(
        input_size=X.shape[1],
        layers_sizes=args.layers,
        learning_rate=args.learning_rate,
        activation=args.activation,
        optimizer=args.optimizer
    )

    history = model.fit(
        X, y, y_double,
        X_val, y_val,
        args.epochs, args.batch_size
        )

    loss_plot(len(history['loss']), history['loss'], history['val_loss'])
    accuracy_plot(len(history['acc']), history['acc'], history['val_acc'])
    model.export('data/model_weights.npz')


if __name__ == "__main__":
    main()
