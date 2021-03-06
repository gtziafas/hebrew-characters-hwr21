from ..types import *
from ..utils import crop_boxes_fixed
from ..data.monkbrill_dataset import MonkbrillDataset
from ..models.cnn import default_cnn_monkbrill, collate
from ..models.loss import FuzzyLoss, TaylorSoftmax
from ..models.training import Trainer, Metrics

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split
from sklearn.model_selection import KFold

ROOT_FOLDER = './data/monkbrill_splits'

FIXED_SHAPE = (75, 75)

# reproducability
SEED = torch.manual_seed(14)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(14)


def main(data_root: str,
         batch_size: int, 
         lr: float,  
         num_epochs: int,
         kfold: int,
         wd: float,
         device: str,
         early_stopping: Maybe[int],
         test_root: Maybe[str],
         save_path: Maybe[str],
         load_path: Maybe[str],
         print_log: bool
        ):

    # an independent function to init a model and train over some epochs for a given train-dev(-test) split
    def train(train_ds: List[Character], dev_ds: List[Character], test_ds: Maybe[List[Character]]=None, 
            save_path: Maybe[str] = None, index: Maybe[int]=None) -> Metrics:
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, worker_init_fn=SEED, collate_fn=collate(device))
        dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=batch_size, worker_init_fn=SEED, collate_fn=collate(device))

        # optionally test in separate split, given from a path directory as argument
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, collate_fn=collate(device)) if test_ds is not None else None

        model = default_cnn_monkbrill().to(device)
        if load_path is not None:
            model.load_pretrained(load_path)
        optim = AdamW(model.parameters(), lr=lr, weight_decay=wd)
        #criterion = CrossEntropyLoss(reduction='mean')
        criterion = FuzzyLoss(num_classes=27, mass_redistribution=0.05)#, softmax=TaylorSoftmax(order=4))
        trainer = Trainer(model, (train_dl, dev_dl, test_dl), optim, criterion, target_metric="accuracy", early_stopping=early_stopping)
        
        if index is not None and save_path is not None:
            save_path = save_path.split('.')[0] + '_' + str(index) + '.p'
        return trainer.iterate(num_epochs, with_save=save_path, print_log=print_log)


    print('Loading / Preprocessing dataset...')
    train_ds = MonkbrillDataset(data_root + '/train', with_preproc=crop_boxes_fixed(FIXED_SHAPE))
    dev_ds = MonkbrillDataset(data_root + '/dev', with_preproc=crop_boxes_fixed(FIXED_SHAPE))
    test_ds = MonkbrillDataset(test_root, with_preproc=crop_boxes_fixed(FIXED_SHAPE)) if test_root is not None else None

    if not kfold:
        # train once in a fixed train-dev split
        print('Training on fixed train-dev splits...')
        best = train(train_ds, dev_ds, test_ds, save_path, None)
        print(f'Results splits: {best}')

    else:
        # k-fold cross validation 
        ds = train_ds.dataset + dev_ds.dataset
        _kfold = KFold(n_splits=kfold, shuffle=True, random_state=14).split(ds)
        accu = 0.
        print(f'{kfold}-fold cross validation...')
        for iteration, (train_idces, dev_idces) in enumerate(_kfold):
            train_ds = [s for i, s in enumerate(ds) if i in train_idces]
            dev_ds = [s for i, s in enumerate(ds) if i in dev_idces]
            best = train(train_ds, dev_ds, test_ds, save_path=save_path, index=iteration)
            print(f'Results {kfold}-fold, iteration {iteration+1}: {best}')
            accu += best['accuracy']
        print(f'Average accuracy {kfold}-fold: {accu/kfold}')



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--data_root', help='path to the directory of training data', type=str, default=ROOT_FOLDER)
    parser.add_argument('-tst', '--test_root', help='path to the directory of testing data (default no test)', type=str, default=None)
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=64)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=15)
    parser.add_argument('-s', '--save_path', help='full path to save best model (default no save)', type=str, default=None)
    parser.add_argument('-wd', '--wd', help='weight decay to use for regularization', type=float, default=1e-02)
    parser.add_argument('-lr', '--lr', help='learning rate to use for optimization', type=float, default=1e-03)
    parser.add_argument('-early', '--early_stopping', help='early stop patience (default no early stopping)', type=int, default=None)
    parser.add_argument('-kfold', '--kfold', help='k-fold cross validation', type=int, default=0)
    parser.add_argument('--print_log', action='store_true', help='print training logs', default=False)
    parser.add_argument('-l', '--load_path', help='full path to load pretrained model (default no load)', type=str, default=None)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)


# we must experiment with the following hyper-params (10-fold):
#
# dropout_rates (in cnn.py):
#       [0.1, 0.1], [0.1, 0.2], *[0.1, 0.25]*, [0.1, 0.33]
#
# learning rates:
#       1e-04, 5e-04, *1e-03*, 5e-03
#
# weight decays:
#       0.001, 0.005, *0.01*, 0.05, 0.1
#
# batch size:
#       32, *64*, 128
#
# e.g: python3 -m qumran_seagulls.scripts.train_cnn_monkbrill -e 15 -early 2 -bs 64 -lr 0.001 -wd 0.01 -kfold 10