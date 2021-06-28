from qumran_seagulls.types import *
from qumran_seagulls.utils import *
from qumran_seagulls.preprocess.char_segm.char_segm import *
from qumran_seagulls.models.viterbi import Viterbi
import torch

import os
import tqdm as tqdm


def main(path: str, device: str = 'cpu'):
    lines = [cv2.imread(os.path.join(path, f), 0) for f in os.listdir(path) if f.endswith('jpg')]
    CS = CharacterSegmenter('checkpoints/cnn_labels_fuzzy05.p', 'cpu')

    for line in lines:
        CS.debug(line)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to the directory of test data', type=str)

    kwargs = vars(parser.parse_args())
    main(**kwargs)
