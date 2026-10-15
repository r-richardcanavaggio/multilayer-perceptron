import numpy as np
from numpy.typing import NDArray
from src.parser import init_parser
from src.plot import accuracy_plot, loss_plot
from src.NeuralNetwork import NeuralNetwork
from src.Math import binary_cross_entropy
from src.EarlyStopping import EarlyStopping
from src.Dataset import Dataset
from src.DataLoader import DataLoader
# from src.History import History
from src.Trainer import Trainer
import matplotlib.pyplot as plt
import pandas


def main():
    np.random.seed(42)

    args = init_parser()

    train_data = Dataset('data/training_data.csv')
    val_data = Dataset('data/testing_data.csv')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    model = NeuralNetwork(
        input_size=train_data.columns,
        layers_sizes=args.layers,
        learning_rate=args.learning_rate,
        activation=args.activation,
        optimizer=args.optimizer
    )

    stop_early = EarlyStopping(args.stop_early) if args.stop_early else None

    trainer = Trainer(model, stop_early, args.epochs)
    history = trainer.fit(train_loader, val_loader)

    M = np.array(history.history['thetas'])
    theta_final = M[-1]
    A = M - theta_final
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    d1 = Vt[0]
    d2 = Vt[1]

    x_traj = np.dot(A, d1)
    y_traj = np.dot(A, d2)
    z_traj = history.history['loss']

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
            Z[i, j] = binary_cross_entropy(train_data.labels, model.forward(train_data.parameters))
    model.set_weights_vector(theta_final)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(u_matrix, v_matrix, Z, cmap='coolwarm', alpha=0.7)

    ax.plot(x_traj, y_traj, z_traj, color='black', marker='o', markersize=4)

    ax.set_xlabel("Composante Principale 1 (d1)")
    ax.set_ylabel("Composante Principale 2 (d2)")
    ax.set_zlabel("Perte (Loss)")

    plt.show()

    loss_plot(history.history['total_epoch'], history.history['loss'], history.history['val_loss'])
    accuracy_plot(history.history['total_epoch'], history.history['accuracy'], history.history['val_accuracy'])
    model.export('data/model_weights.npz')
    pandas.DataFrame(history.history).to_csv('history.csv')


if __name__ == "__main__":
    main()
