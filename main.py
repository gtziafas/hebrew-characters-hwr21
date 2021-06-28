from qumran_seagulls.types import *
from qumran_seagulls.preprocess.line_segm.line_segm_astar import call_lineSeg
from qumran_seagulls.preprocess.char_segm.char_segm import default_char_segm
from qumran_seagulls.models.viterbi import default_viterbi
from qumran_seagulls.models.style_classifier import default_style_classifier
import torch

import os
import cv2
import tqdm as tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(path_to_test_imgs: str):
    # fetch images
    image_files = [f for f in os.listdir(path_to_test_imgs)]
    images = [cv2.imread(os.path.join(path_to_test_imgs, file), 0) for file in image_files]

    # init sub-modules
    CharSegm = default_char_segm(device=DEVICE)
    StyleCLS = default_style_classifier(device=DEVICE)
    Viterbi = default_viterbi(device=DEVICE)

    # pipeline
    all_predictions, all_styles = [], []
    for i, image in enumerate(images):
        print(f'Running for {image_files[i]}:')

        # run line segmentation
        lines = call_lineSeg(image)

        predictions, styles = [], []
        print(f'Performing character recognition...')
        for line in lines:
            all_likelihoods = CharSegm(line)
            best_paths = [Viterbi(torch.flip(lkhd, [0]))[0] for lkhd in all_likelihoods]
            predictions.append(sum([[LABEL_MAP[i] for i in path] for path in best_paths], []))
        all_predictions.append(predictions)
        
        print(f'Performing style classification...')
        all_styles.append(StyleCLS(lines)) 

    # write output
    print(f'Writting output...')
    if not os.path.isdir('./results'):
        os.mkdir('./results')

    for file, preds, style in  zip(image_files, all_predictions, all_styles):
        _characters = './results/' + file.split('.')[0] + '_characters.txt'
        with open(_characters, 'w') as g:
            preds = [' '.join(ps) for ps in preds]
            g.write('\n'.join(preds))

        _style = './results/' + file.split('.')[0] + '_style.txt'
        with open(_style, 'w') as g:
            g.write(style)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])