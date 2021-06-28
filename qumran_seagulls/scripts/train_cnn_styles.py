from ..types import *
from ..models.cnn import *
from ..utils import crop_boxes_fixed, pad_with_frame
from ..data.styles_dataset import StylesDataset
from ..models.training import Trainer, Metrics

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split
from sklearn.model_selection import KFold

ROOT_FOLDER = './data/styles/characters'

FIXED_SHAPE = (75, 75)

# reproducability
SEED = torch.manual_seed(14)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(14)


def collate(device: str, with_padding: Maybe[Tuple[int, int]] = None) -> Callable[[List[Character]], Tuple[Tensor, Tensor]]:
    
    def _collate(batch: List[Character]) -> Tuple[Tensor, Tensor]:
        imgs, styles = zip(*[(s.image, s.style) for s in batch])
    
        # pad to equal size if desired
        if with_padding is not None:
            imgs = pad_with_frame(imgs, desired_shape=with_padding)

        # normalize to [0,1], tensorize, send to device, add channel dimension and stack
        imgs = torch.stack([torch.tensor(img / 0xff, dtype=floatt, device=device) for img in imgs], dim=0).unsqueeze(1)
        styles = torch.stack([torch.tensor(label, dtype=longt, device=device) for label in styles], dim=0)
        return imgs, styles

    return _collate


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
         checkpoint: Maybe[str],
         print_log: bool,
         concat: bool,
         multi: bool
        ):

    collator = collate_concat(device) if concat or multi else collate(device)
    make_model = concat_cnn_styles if concat else multitask_cnn if multi else default_cnn_styles
    target_metric = 'accuracy' if not multi else 'style_accuracy'

    # an independent function to init a model and train over some epochs for a given train-dev(-test) split
    def train(train_ds: List[Character], dev_ds: List[Character], test_ds: Maybe[List[Character]]=None,
                save_path: Maybe[str] = None, index: Maybe[int] = None) -> Metrics:
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, worker_init_fn=SEED, collate_fn=collator)
        dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=batch_size, worker_init_fn=SEED, collate_fn=collator)

        # optionally test in separate split, given from a path directory as argument
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, collate_fn=collator) if test_ds is not None else None

        model = make_model().to(device) 
        if load_path is not None:
            if not multi:
                model.load_pretrained(load_path)
            else:
                model.load_char_model(load_path)

        optim = AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = CrossEntropyLoss(reduction='mean')
        #criterion = FuzzyLoss(num_classes=27, mass_redistribution=0.3)#, softmax=TaylorSoftmax(order=4))
        trainer = Trainer(model, (train_dl, dev_dl, test_dl), optim, criterion, target_metric=target_metric, 
                early_stopping=early_stopping, multitask=multi)
        
        if index is not None and save_path is not None:
            save_path = save_path.split('.')[0] + '_' + str(index) + '.p'
        return trainer.iterate(num_epochs, with_save=save_path, print_log=print_log)


    print('Loading / Preprocessing dataset...')
    if not checkpoint:
        train_ds = StylesDataset(data_root + '/train', with_preproc=crop_boxes_fixed(FIXED_SHAPE))
        dev_ds = StylesDataset(data_root + '/dev', with_preproc=crop_boxes_fixed(FIXED_SHAPE))
        test_ds = StylesDataset(test_root, with_preproc=crop_boxes_fixed(FIXED_SHAPE)) if test_root is not None else None

    else:
        # load splits from checkpoint
        checkpoint = torch.load(checkpoint)
        train_ds = [Character(image=s[0], label=s[1], style=s[2]) for s in checkpoint['train']]
        dev_ds = [Character(image=s[0], label=s[1], style=s[2]) for s in checkpoint['dev']]
        test_ds = [Character(image=s[0], label=s[1], style=s[2]) for s in checkpoint['test']] 


    # train according to given flags
    if kfold and checkpoint:
        raise ValueError('checkpoint and kfold flags cannot be True at the same time...')

    elif not kfold:
        # train once in a 80%-20% train-dev split
        print('Training on fixed train-dev split...')
        best = train(train_ds, dev_ds, test_ds)
        print(f'Results split: {best}')

    else:
        # k-fold cross validation 
        ds = train_ds.dataset + dev_ds.dataset
        _kfold = KFold(n_splits=kfold, shuffle=True, random_state=14).split(ds)
        accu = 0.
        print(f'{kfold}-fold cross validation...')
        for iteration, (train_idces, dev_idces) in enumerate(_kfold):
            train_ds = [s for i, s in enumerate(ds) if i in train_idces]
            dev_ds = [s for i, s in enumerate(ds) if i in dev_idces]
            best = train(train_ds, dev_ds, test_ds, save_path, iteration)
            print(f'Results {kfold}-fold, iteration {iteration+1}: {best}')
            accu += best[target_metric]
        print(f'Average accuracy {kfold}-fold: {accu/kfold}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--data_root', help='path to the directory of training data', type=str, default=ROOT_FOLDER)
    parser.add_argument('-tst', '--test_root', help='path to the directory of testing data (default no test)', type=str, default=None)
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=16)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=15)
    parser.add_argument('-s', '--save_path', help='full path to save best model (default no save)', type=str, default=None)
    parser.add_argument('-wd', '--wd', help='weight decay to use for regularization', type=float, default=1e-02)
    parser.add_argument('-lr', '--lr', help='learning rate to use for optimization', type=float, default=1e-03)
    parser.add_argument('-early', '--early_stopping', help='early stop patience (default no early stopping)', type=int, default=None)
    parser.add_argument('-kfold', '--kfold', help='k-fold cross validation', type=int, default=0)
    parser.add_argument('--print_log', action='store_true', help='print training logs', default=False)
    parser.add_argument('-l', '--load_path', help='full path to load pretrained model (default no load)', type=str, default=None)
    parser.add_argument('-chp', '--checkpoint', help='whether to use given file to load data', type=str, default=None)
    parser.add_argument('--concat', action='store_true', help='concatenate character labels for prediction', default=False)
    parser.add_argument('--multi', action='store_true', help='multi-task cnn', default=False)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)

# best dev:{'epoch': 18, 'loss': 0.86244, 'accuracy': 0.6225}, @test: {'loss': 0.90476, 'accuracy': 0.5828}