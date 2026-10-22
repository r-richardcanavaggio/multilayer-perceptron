from src.parser import init_parser
from src.NeuralNetwork import NeuralNetwork
from src.EarlyStopping import EarlyStopping
from src.Dataset import Dataset
from src.DataLoader import DataLoader
from src.Trainer import Trainer
from src.LossLandscape import LossLandscape
from src.StandardScaler import StandardScaler
import pandas


def main() -> None:
    args = init_parser()

    train_data = Dataset('data/training_data.csv')
    val_data = Dataset('data/testing_data.csv')

    scaler = StandardScaler()
    scaler.fit(train_data.parameters)

    train_data.normalize(scaler)
    val_data.normalize(scaler)

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size, shuffle=False)

    model = NeuralNetwork(
        input_size=train_data.columns,
        layers_sizes=args.layers,
        learning_rate=args.learning_rate,
        activation=args.activation,
        optimizer=args.optimizer,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.eps
    )

    stop_early = EarlyStopping(args.stop_early) if args.stop_early else None

    trainer = Trainer(model, stop_early, args.epochs)
    history = trainer.fit(train_loader, val_loader)

    loss3d = LossLandscape(model, train_data,
                           history.history['thetas'], history.history['loss'])
    loss3d.plot()

    history.plot_loss()
    history.plot_acc()
    model.export('data/model_weights.npz', scaler=scaler)
    pandas.DataFrame(history.history).to_csv('history.csv')


if __name__ == "__main__":
    main()
