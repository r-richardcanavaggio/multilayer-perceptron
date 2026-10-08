import matplotlib.pyplot as plt


def loss_plot(epochs: int, losses_train: list[float], losses_val: list[float]):
    x = [i for i in range(epochs)]
    plt.plot(x, losses_train, color='blue', label='training loss')
    plt.plot(x, losses_val,
             color='orange', linestyle='--',
             label='validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def accuracy_plot(
        epochs: int, accuracy_train: list[float],
        accuracy_val: list[float]
        ) -> None:
    x = [i for i in range(epochs)]
    plt.plot(x, accuracy_train, color='blue', label='training accuracy')
    plt.plot(x, accuracy_val,
             color='orange', linestyle='--',
             label='validation accuracy'
             )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
