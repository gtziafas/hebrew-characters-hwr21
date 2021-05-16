from ..types import *

import torch
import torch.nn as nn 

Metrics = Dict[str, Any]


def train_epoch(model: nn.Module, dl: DataLoader, optim: Optimizer, criterion: nn.Module) -> Metrics: 
    model.train()
    epoch_loss, total_correct = 0., 0
    for batch_idx, (x, y) in enumerate(dl):
        # forward
        preds = model.forward(x)
        loss = criterion(preds, y)

        # backprop
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        # update
        epoch_loss += loss.item()
        total_correct += (preds.argmax(dim=-1) == y).sum().item()

    epoch_loss /= len(dl)
    accuracy = total_correct / len(dl.dataset)
    return {'loss': round(epoch_loss, 5), 'accuracy': round(accuracy, 4)}


@torch.no_grad()
def eval_epoch(model: nn.Module, dl: DataLoader, criterion: nn.Module) -> Metrics: 
    model.eval()
    epoch_loss, total_correct = 0., 0
    for x, y in dl:
        # forward
        preds = model.forward(x)
        loss = criterion(preds, y)

        # update
        epoch_loss += loss.item()
        total_correct += (preds.argmax(dim=-1) == y).sum().item()

    epoch_loss /= len(dl)
    accuracy = total_correct / len(dl.dataset)
    return {'loss': round(epoch_loss, 5), 'accuracy': round(accuracy, 4)}


class Trainer(ABC):
    def __init__(self, 
            model: nn.Module, 
            dls: Tuple[DataLoader, ...],
            optimizer: Optimizer, 
            criterion: nn.Module, 
            target_metric: str,
            early_stopping: Maybe[int] = None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dl, self.dev_dl, self.test_dl = dls
        self.logs = {'train': [], 'dev': [], 'test': []}
        self.target_metric = target_metric
        self.trained_epochs = 0
        self.early_stop_patience = early_stopping

    def iterate(self, num_epochs: int, print_log: bool = False, with_save: Maybe[str] = None) -> Metrics:
        best = {self.target_metric: 0.}
        patience = self.early_stop_patience if self.early_stop_patience is not None else num_epochs
        for epoch in range(num_epochs):
            self.step(print_log)

            # update logger for best - save - test - early stopping
            if self.logs['dev'][-1][self.target_metric] > best[self.target_metric]:
                best = self.logs['dev'][-1]
                patience = self.early_stop_patience if self.early_stop_patience is not None else num_epochs

                if with_save is not None:
                    torch.save(self.model.state_dict(), with_save)

                if self.test_dl is not None:
                    self.logs['test'].append({'epoch': epoch+1, **eval_epoch(self.model, self.test_dl, self.criterion)})

            else:
                patience -= 1
                if not patience:
                    self.trained_epochs += epoch + 1
                    return best

        self.trained_epochs += num_epochs
        return best

    def step(self, print_log: bool = False):
        current_epoch = len(self.logs['train']) + 1

        # train - eval this epoch
        self.logs['train'].append({'epoch': current_epoch, **train_epoch(self.model, self.train_dl, self.optimizer, self.criterion)})
        self.logs['dev'].append({'epoch': current_epoch, **eval_epoch(self.model, self.dev_dl, self.criterion)})
        
        # print if wanted
        if print_log:
            print('TRAIN:')
            for k,v in self.logs['train'][-1].items():
                print(f'{k} : {v}')
            print()
            print('DEV:')
            for k,v in self.logs['dev'][-1].items():
                print(f'{k} : {v}')
            print('==' * 72)
