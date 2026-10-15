class EarlyStopping:
    def __init__(self, patience: int=10):
        self.best_val_loss = float('inf')
        self.count_loss = 0
        self.patience = patience

    def should_stop(self, current_val_loss: float) -> bool:
        if current_val_loss < self.best_val_loss:
            self.count_loss = 0
            self.best_val_loss = current_val_loss
            return False
        elif current_val_loss >= self.best_val_loss and self.count_loss != self.patience:
            self.count_loss += 1
            return False
        else:
            return True
