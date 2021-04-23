from types import *
from data_loader import MonkbrillDataset
from training import Trainer, Metrics

import torch.nn as nn 
from torch import tensor, stack, manual_seed
from torch.optim import AdamW, Adam
from torch.utils.data import random_split
from sklearn.model_selection import KFold

manual_seed(14)


class BaselineCNN(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.Dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        ...


def collate(batch: List[Character], ..., device: str) -> Tuple[Tensor, Tensor]:
    xs, ys = zip(*[(s.image, s.label) for s in batch])
    # sth to do with variable image sizes (look at resolu_distr.png)
    ys = stack([tensor(y, dtype=longt, device=device) for y in ys], dim=0)

    

def main(data_root: str,
         batch_size: int, 
         lr: float, 
         wd: float, 
         dropout: float, 
         num_epochs: int,
         early_stopping: int,
         kfold: bool,
         device: str,
         print_log: bool):

    # an independent function to init a model and train over some epochs
    # for a given train-dev split
    def train(train_ds: List[Character], dev_ds: List[Character]) -> Metrics:

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size,
            collate_fn = lambda b: collate(b, ..., device))
        dev_dl = DataLoader(train_ds, shuffle=False, batch_size=batch_size,
            collate_fn = lambda b: collate(b, ..., device))

        model = BaselineCNN().to(device)
        optim = AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss(reduction='mean')

        trainer = Trainer(model, (train_dl, dev_dl), optim, criterion, target_metric="accuracy", early_stopping=early_stopping)

        return trainer.iterate(num_epochs)


    ds = MonkbrillDataset(data_root)

    if not kfold:
        # train once in a 80%-20% train-dev split
        dev_size = int(.2 * len(ds))
        train_ds, dev_ds = random_split(ds, [len(ds) - dev_size, dev_size])
        best = train(train_ds, dev_ds)
        print(f'Results 80-20 random split: {best}')

    else:
        # k-fold cross validation 
        _kfold = KFold(n_splits=kfold, shuffle=True, random_state=14).split(ds)
        accu = 0.
        for iteration, (train_idces, dev_idces) in enumerate(_kfold):
            train_ds = [s for i, s in enumerate(ds) if i in train_idces]
            dev_ds = [s for i, s in enumerate(ds) if i in dev_idces]
            best = train(train_ds, dev_ds)
            print(f'Results {kfold}-fold, iteration {iteration+1}: {best}')
            accu += best['accuracy']
        print(f'Average accuracy {kfold}-fold: {accu/kfold}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--data_root', help='path to the directory of training data', type=str)
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=16)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=7)
    #parser.add_argument('-s', '--save_path', help='where to save best model', type=str, default=f'{SAVE_PREFIX}/COVID-19-event/checkpoints')
    parser.add_argument('-dr', '--dropout', help='model dropout to use in training', type=float, default=0.5)
    parser.add_argument('-wd', '--weight_decay', help='weight decay to use for regularization', type=float, default=0.)
    parser.add_argument('-lr', '--lr', help='learning rate to use for optimization', type=float, default=1e-03)
    parser.add_argument('-early', '--early_stopping', help='early stop patience (default no)', type=int, default=0)
    parser.add_argument('-kfold', '--kfold', help='k-fold cross validation', type=int, default=0)
    parser.add_argument('--print_log', action='store_true', help='print training logs', default=False)

    kwargs = vars(parser.parse_args())
    main(**kwargs)