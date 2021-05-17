from ..types import * 
from ..utils import pad_with_frame
from ..models.loss import TaylorSoftmax
from ..models.cnn import load_pretrained, collate

import torch
import torch.nn as nn
from matplotlib import pyplot as plt 
import pickle

ROOT_FOLDER = 'checkpoints/monkbrill_splits_seed14.p'
FIXED_SHAPE = (75, 75)

COLORS_MAP = {
    0 : 'bo--',
    1 : 'ro--',
    2 : 'go--',
    3 : 'mo--',
    4 : 'yo--',
    5 : 'ko--',
    6 : 'co--'
}


@torch.no_grad()
def main(data_root: str, model_paths: str):
    model_paths = model_paths.split(',')

    # load dataset from binary file provided
    chp = torch.load(data_root)
    #train_ds = [Character(image=i, label=l) for i, l in chp['train']]
    samples = [Character(image=i, label=l) for i, l in chp['dev']]
    images = pad_with_frame([s.image for s in samples], FIXED_SHAPE)
    tensors = torch.stack([torch.tensor(i/0xff, dtype=floatt) for i in images]).unsqueeze(1)
 
    keys = [p.split('/')[1].split('_')[0] for p in model_paths]
    methods = [TaylorSoftmax(int(k[-1])) if 'taylor' in k else nn.Softmax(-1) for k in keys]
    all_scores = []
    print('Computing scores for different models...\n')
    for path, method in zip(model_paths, methods):
        model = load_pretrained(path)
        all_scores.append(method(model.forward(tensors)))

    print('Truth\t' + '\t'.join(keys))

    for sid, s in enumerate(samples):
        all_preds = [s.label_str] + [LABEL_MAP[sc[sid].argmax(-1).item()] for sc in all_scores]
        if len(set(all_preds)) == 1:
            continue
        print('\t'.join(all_preds))
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.imshow(s.image)
        for idx, sc in enumerate(all_scores):
            color_str = COLORS_MAP[idx]
            ax2.plot(sc[sid], color_str, label=keys[idx])
        ax2.grid(True)
        ax2.legend()
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--data_root', help='path to the directory of training data', type=str, default=ROOT_FOLDER)
    parser.add_argument('-p', '--model_paths', help='full paths to pretrained models to compare (seperated by ,)', type=str)

    kwargs = vars(parser.parse_args())
    main(**kwargs)