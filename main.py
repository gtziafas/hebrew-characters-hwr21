from qumran_seagulls.types import *
from qumran_seagulls.preprocess.line_segm.line_segm_astar import call_lineSeg
from qumran_seagulls.preprocess.char_segm.char_segm import CharacterSegmenter
from qumran_seagulls.models.viterbi import default_viterbi
from qumran_seagulls.models.style_classifier import default_style_classifier
import torch

import os
import cv2
import tqdm as tqdm


CNN_RECO_PATH = 'checkpoints/cnn_labels_augm_fuzzy.p'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(path_to_test_imgs: str):
    # fetch images
    image_files = [f for f in os.listdir(path_to_test_imgs)]
    images = [cv2.imread(os.path.join(path_to_test_imgs, file), 0) for file in image_files]

    # init sub-modules
    CharSegm = CharacterSegmenter(load_path=CNN_RECO_PATH, device=DEVICE)
    StyleCLS = default_style_classifier(device=DEVICE)
    Viterbi = default_viterbi(device=DEVICE)

    # pipeline
    all_predictions, all_styles = [], []
    for i, image in enumerate(images):
        print(f'Running for {image_files[i]}:')

        # run line segmentation
        lines = call_lineSeg(image)

        predictions = []
        styles = []
        for line in lines:
            print(f'Performing character segmentation and recognition...')
            all_likelihoods = CharSegm(line)
            best_paths = [Viterbi(torch.flip(lkhd, [0]))[0] for lkhd in all_likelihoods]
            predictions.append(sum([[LABEL_MAP[i] for i in path] for path in best_paths], []))
        all_predictions.append(predictions)
        
        print(f'Performing style classification...')
        all_styles.append(StyleCLS(lines))
        #all_styles.append('Archaic')

    # write output
    print(f'Writting output...')
    if not os.path.isdir('./results'):
        os.mkdir('./results')

    for file, preds, style in  zip(image_files, all_predictions, all_styles):
        _characters = os.path.join('./results', file.split('.'), '_characters.txt')
        with open(_characters, 'w') as g:
            g.write('\n'.join(preds))

        _style = os.path.join('./results', file.split('.'), '_style.txt')
        with open(_style, 'w') as g:
            g.write(style)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])