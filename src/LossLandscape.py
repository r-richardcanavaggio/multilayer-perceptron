from src.NeuralNetwork import NeuralNetwork
from src.Dataset import Dataset
from numpy.typing import NDArray
from numpy import float64
import numpy as np
from src.Math import binary_cross_entropy
import matplotlib.pyplot as plt


class LossLandscape:
    def __init__(self, model: NeuralNetwork, dataset: Dataset,
                 thetas: list[NDArray[float64]], losses: list[float]):
        self.model = model
        self.dataset = dataset
        self.thetas = thetas
        self.losses = losses

        self.d1: NDArray[float64]
        self.d2: NDArray[float64]

        self.x_traj: NDArray[float64]
        self.y_traj: NDArray[float64]
        self.z_traj: list[float]

    def _compute_pca_directions(self):
        matrix = np.array(self.thetas)
        self.theta_final = matrix[-1]
        centered_matrix = matrix - self.theta_final

        U, S, Vt = np.linalg.svd(centered_matrix, full_matrices=False)
        self.d1 = Vt[0]
        self.d2 = Vt[1]

        self.x_traj = np.dot(centered_matrix, self.d1)
        self.y_traj = np.dot(centered_matrix, self.d2)
        self.z_traj = self.losses

    def _compute_surface(self, resolution: int):
        u_vals: NDArray[np.float64] = np.linspace(
            self.x_traj.min() - 0.5, self.x_traj.max() + 0.5,
            resolution, dtype=np.float64)
        v_vals: NDArray[np.float64] = np.linspace(
            self.y_traj.min() - 0.5, self.y_traj.max() + 0.5,
            resolution, dtype=np.float64)

        self.u_matrix, self.v_matrix = np.meshgrid(u_vals, v_vals)

        self.Z = np.zeros((resolution, resolution))

        for i in range(resolution):
            for j in range(resolution):
                u = self.u_matrix[i, j]
                v = self.v_matrix[i, j]

                theta = self.theta_final + (u * self.d1) + (v * self.d2)
                self.model.set_weights_vector(theta)
                self.Z[i, j] = binary_cross_entropy(
                    self.dataset.labels,
                    self.model.forward(self.dataset.parameters))

        self.model.set_weights_vector(self.theta_final)

    def plot(self, resolution=30):
        self._compute_pca_directions()
        self._compute_surface(resolution)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(self.u_matrix, self.v_matrix, self.Z,
                        cmap='coolwarm', alpha=0.7)

        ax.plot(self.x_traj, self.y_traj, self.z_traj,
                color='black', marker='o', markersize=4)

        ax.set_xlabel("Composante Principale 1 (d1)")
        ax.set_ylabel("Composante Principale 2 (d2)")
        ax.set_zlabel("Perte (Loss)")

        plt.show()
