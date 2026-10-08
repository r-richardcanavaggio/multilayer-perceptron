import numpy as np
import pandas
from src.NeuralNetwork import init_nn
from src.parser import init_parser
from src.plot import accuracy_plot, loss_plot


def load_data(train_path: str, val_path: str):
    training_data = pandas.read_csv(train_path).to_numpy()
    validation_data = pandas.read_csv(val_path).to_numpy()

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
    X, y, y_double, X_val, y_val = load_data('data/training_data.csv', 'data/testing_data.csv')

    model = init_nn(X.shape[1], args.layers,
                    args.activation, args.learning_rate, args.optimizer)

    history = model.fit(X, y, y_double, X_val, y_val, args.epochs, args.batch_size)

    loss_plot(args.epochs, history['loss'], history['val_loss'])
    accuracy_plot(args.epochs, history['acc'], history['val_acc'])
    model.export()


if __name__ == "__main__":
    main()
