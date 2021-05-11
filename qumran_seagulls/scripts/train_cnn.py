from ..types import *
from ..utils import crop_boxes_dynamic, filter_large
from ..data.monkbrill_dataset import MonkbrillDataset
from ..models.cnn import default_cnn, collate
from ..models.training import Trainer, Metrics

from torch import manual_seed
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split
from sklearn.model_selection import KFold

manual_seed(14)

ROOT_FOLDER = './data/monkbrill'

FIXED_SHAPE = (75, 75)


def main(data_root: str,
         batch_size: int, 
         lr: float, 
         #dropout: float, 
         num_epochs: int,
         early_stopping: int,
         kfold: int,
         wd: float,
         device: str,
         print_log: bool):

    # an independent function to init a model and train over some epochs for a given train-dev split
    def train(train_ds: List[Character], dev_ds: List[Character]) -> Metrics:
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=collate(device, FIXED_SHAPE))
        dev_dl = DataLoader(train_ds, shuffle=False, batch_size=batch_size, collate_fn=collate(device, FIXED_SHAPE))

        model = default_cnn().to(device)
        optim = AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = CrossEntropyLoss(reduction='mean')

        trainer = Trainer(model, (train_dl, dev_dl), optim, criterion, target_metric="accuracy", \
            print_log=print_log, early_stopping=early_stopping)

        return trainer.iterate(num_epochs)


    print('Loading / Preprocessing dataset...')
    ds = MonkbrillDataset(data_root, with_preproc=filter_large(FIXED_SHAPE))

    if not kfold:
        # train once in a 80%-20% train-dev split
        dev_size = int(.2 * len(ds))
        train_ds, dev_ds = random_split(ds, [len(ds) - dev_size, dev_size])
        print('Training on random train-dev split...')
        best = train(train_ds, dev_ds)
        print(f'Results random split: {best}')

    else:
        # k-fold cross validation 
        _kfold = KFold(n_splits=kfold, shuffle=True, random_state=14).split(ds)
        accu = 0.
        print(f'{kfold}-fold cross validation...')
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
    parser.add_argument('-r', '--data_root', help='path to the directory of training data', type=str, default=ROOT_FOLDER)
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=64)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=20)
    #parser.add_argument('-s', '--save_path', help='where to save best model', type=str, default=f'{SAVE_PREFIX}/COVID-19-event/checkpoints')
    #parser.add_argument('-dr', '--dropout', help='model dropout to use in training', type=float, default=0.5)
    parser.add_argument('-wd', '--wd', help='weight decay to use for regularization', type=float, default=0.)
    parser.add_argument('-lr', '--lr', help='learning rate to use for optimization', type=float, default=1e-03)
    parser.add_argument('-early', '--early_stopping', help='early stop patience (default no)', type=int, default=0)
    parser.add_argument('-kfold', '--kfold', help='k-fold cross validation', type=int, default=0)
    parser.add_argument('--print_log', action='store_true', help='print training logs', default=False)

    kwargs = vars(parser.parse_args())
    main(**kwargs)