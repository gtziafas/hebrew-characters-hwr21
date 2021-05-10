from ..types import *
from ..utils import tighten_boxes
from ..data.monkbrill_dataset import MonkbrillDataset
from ..models.training import Trainer, Metrics

import cv2
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
from torch import tensor, stack, manual_seed
from torch.optim import AdamW, Adam
from torch.utils.data import random_split
from sklearn.model_selection import KFold

manual_seed(14)

ROOT_FOLDER = './data/monkbrill'
FIXED_SHAPE = (70, 65)


class BaselineCNN(nn.Module):
    def __init__(self,
                 num_classes: int, 
                 dropout_rates: List[float]
                ):
        super().__init__()
        self.dropout_rates = dropout_rates
        self.block1 = self.block(in_channels=1, out_channels=16, conv_kernel=3, pool_kernel=3) 
        self.block2 = self.block(in_channels=16, out_channels=32, conv_kernel=3, pool_kernel=2)
        self.block3 = self.block(in_channels=32, out_channels=64, conv_kernel=3, pool_kernel=2)
        self.cls = nn.Linear(in_features=768, out_features=num_classes)


    def block(self, in_channels: int, out_channels: int, conv_kernel: int, pool_kernel: int, conv_stride: int=1):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, stride=conv_stride),
                             nn.GELU(),
                             nn.MaxPool2d(kernel_size=pool_kernel)
                            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = F.dropout(x, p = self.dropout_rates[0])
        x = self.block3(x)
        x = F.dropout(x, p = self.dropout_rates[1])
        x = x.flatten(1)
        return self.cls(x)


def pad_imgs_in_batch(batch: List[array], desired_shape: Tuple[int, int]) -> List[array]:
    shapes = [img.shape for img in batch]
    
    # # identify maximum resolution in batch
    # h_max, w_max = max([s[0] for s in shapes]), max([s[1] for s in shapes])
    # center = (h_max // 2, w_max // 2)

    # have fixed resolution frame
    center = (desired_shape[0] // 2, desired_shape[1] // 2)

    # paste every image in the center of a desired resolution frame to pad all images
    offsets = [(center[0] - shape[0] // 2, center[1] - shape[1] // 2) for shape in shapes]
    batch_padded = []
    for img, shape, offset in zip(batch, shapes, offsets):
        batch_padded.append(np.zeros((desired_shape[0], desired_shape[1])))
        batch_padded[-1][offset[0] : offset[0] + shape[0], offset[1] : offset[1] + shape[1]] = img 
    return batch_padded


def collate(batch: List[Character], device: str) -> Tuple[Tensor, Tensor]:
    imgs, labels = zip(*[(s.image, s.label) for s in batch])
    
    # pad to equal size
    imgs = pad_imgs_in_batch(imgs, desired_shape=FIXED_SHAPE)

    # normalize images to [0, 1] range and add channel dimension
    imgs = [np.expand_dims((img / 0xff).astype(np.float), axis=0) for img in imgs]

    # tensorize, send to device and stack
    imgs = stack([tensor(img, dtype=floatt, device=device) for img in imgs], dim=0)
    labels = stack([tensor(label, dtype=longt, device=device) for label in labels], dim=0)
    return imgs, labels
    

def main(data_root: str,
         batch_size: int, 
         lr: float, 
         #dropout: float, 
         num_epochs: int,
         early_stopping: int,
         kfold: bool,
         wd: float,
         device: str,
         print_log: bool):

    # an independent function to init a model and train over some epochs
    # for a given train-dev split
    def train(train_ds: List[Character], dev_ds: List[Character]) -> Metrics:

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size,
            collate_fn = lambda b: collate(b, device))
        dev_dl = DataLoader(train_ds, shuffle=False, batch_size=batch_size,
            collate_fn = lambda b: collate(b, device))

        model = BaselineCNN(num_classes=27, dropout_rates=[0, 0]).to(device)
        optim = AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss(reduction='mean')

        trainer = Trainer(model, (train_dl, dev_dl), optim, criterion, target_metric="accuracy", \
            print_log=print_log, early_stopping=early_stopping)

        return trainer.iterate(num_epochs)


    print('Loading / Preprocessing dataset...')
    ds = MonkbrillDataset(data_root, tighten_boxes)

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