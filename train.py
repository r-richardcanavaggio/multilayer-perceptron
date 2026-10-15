import numpy as np
from src.parser import init_parser
from src.NeuralNetwork import NeuralNetwork
from src.EarlyStopping import EarlyStopping
from src.Dataset import Dataset
from src.DataLoader import DataLoader
from src.Trainer import Trainer
from src.LossLandscape import LossLandscape
import pandas


def main():
    np.random.seed(42)

    args = init_parser()

    train_data = Dataset('data/training_data.csv')
    val_data = Dataset('data/testing_data.csv')

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size, shuffle=False)

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

    loss3d = LossLandscape(model, train_data,
                           history.history['thetas'], history.history['loss'])
    loss3d.plot()

    history.plot_loss()
    history.plot_acc()
    model.export('data/model_weights.npz')
    pandas.DataFrame(history.history).to_csv('history.csv')


if __name__ == "__main__":
    main()
