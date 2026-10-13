import numpy as np
from numpy.typing import NDArray
from src.parser import init_parser
from src.plot import accuracy_plot, loss_plot
from src.NeuralNetwork import NeuralNetwork
from src.ft_load import ft_load
from src.Math import binary_cross_entropy
import matplotlib.pyplot as plt
import pandas


def load_data(train_path: str, val_path: str):
    training_data = ft_load(train_path).to_numpy()
    validation_data = ft_load(val_path).to_numpy()

    # First column of training_data, real output
    y_train = training_data[:, 0]
    y_val = validation_data[:, 0]

    X_train = np.ascontiguousarray(training_data[:, 1:])
    X_val = np.ascontiguousarray(validation_data[:, 1:])

    return X_train, y_train, X_val, y_val


def main():
    np.random.seed(42)

    args = init_parser()
    X, y, X_val, y_val = load_data(
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
        X, y,
        X_val, y_val,
        args.epochs, args.batch_size,
        args.stop_early
    )

    M = np.array(history['thetas'])
    theta_final = M[-1]
    A = M - theta_final
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    d1 = Vt[0]
    d2 = Vt[1]

    x_traj = np.dot(A, d1)
    y_traj = np.dot(A, d2)
    z_traj = history['loss']

    u_vals: NDArray[np.float64] = np.linspace(x_traj.min() - 0.5, x_traj.max() + 0.5, 30, dtype=np.float64)
    v_vals: NDArray[np.float64] = np.linspace(y_traj.min() - 0.5, y_traj.max() + 0.5, 30, dtype=np.float64)

    u_matrix, v_matrix = np.meshgrid(u_vals, v_vals)

    Z = np.zeros((30, 30))

    for i in range(30):
        for j in range(30):
            u = u_matrix[i, j]
            v = v_matrix[i, j]

            theta = theta_final + (u * d1) + (v * d2)
            model.set_weights_vector(theta)
            perte = binary_cross_entropy(y, model.forward(X))
            Z[i, j] = perte
    model.set_weights_vector(theta_final)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(u_matrix, v_matrix, Z, cmap='coolwarm', alpha=0.7)

    ax.plot(x_traj, y_traj, z_traj, color='black', marker='o', markersize=4)

    ax.set_xlabel("Composante Principale 1 (d1)")
    ax.set_ylabel("Composante Principale 2 (d2)")
    ax.set_zlabel("Perte (Loss)")

    plt.show()

    loss_plot(history['total_epoch'], history['loss'], history['val_loss'])
    accuracy_plot(history['total_epoch'], history['acc'], history['val_acc'])
    model.export('data/model_weights.npz')
    pandas.DataFrame(history).to_csv('history.csv')


if __name__ == "__main__":
    main()
