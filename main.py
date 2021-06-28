from qumran_seagulls.types import *
from qumran_seagulls.preprocess.line_segm.line_segm_astar import call_lineSeg
from qumran_seagulls.preprocess.char_segm.char_segm import CharacterSegmenter
from qumran_seagulls.models.viterbi import Viterbi
from qumran_seagulls.models.style_classifier import StyleClassifier
import torch

import os
import tqdm as tqdm


CNN_CHARACTERS_PATH = 'checkpoints/cnn_labels_fuzzy05.p'
CNN_STYLES_PATH = 'checkpoints/cnn_styles.p'
TRANS_MATRIX_PATH = 'checkpoints/trans.p'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(path_to_test_imgs: str):
    # fetch images
    image_files = [f for f in os.listdir(path_to_test_imgs) if 'binari' in f]
    images = [cv2.imread(os.path.join(path_to_test_imgs, file), 0) for file in image_files]

    # init sub-modules
    CharSegm = CharacterSegmenter(load_path=CNN_CHARACTERS_PATH, device=DEVICE)
    #StyleCLS = StyleClassifier(load_path=CNN_STLES_PATH, device=device)
    Viterbi = Viterbi(num_classes=27, transition_matrix=torch.load(TRANS_MATRIX_PATH))

    # pipeline
    all_predictions, all_styles = [], []
    for i, image in enumerate(images):
        print(f'Running {image_paths[i].spliT("/")[1]}')

        # run line segmentation
        lines = call_lineSeg(image)

        predictions = []
        styles = []
        for line in lines:
            print(f'Performing character segmentation and recognition...')
            all_likelihoods, crops = CharSegm(line)
            best_paths = [Viterbi(lkhd)[0] for lkhd in all_likelihoods]
            predictions.append(sum([[LABEL_MAP[i] for i in path] for path in best_paths], []))
        all_predictions.append(predictions)
        
        print(f'Performing style classification...')
        #all_styles.append(StyleCLS(lines))
        all_styles.append('Archaic')
        
    # write output
    if not os.path.isdir('./results'):
        os.mkdir('./results')

    for file, preds, style in  zip(image_files, preds, styles):
        _characters = os.path.join('./results', file.split('.'), '_characters.txt')
        with open(_characters, 'w') as g:
            g.write('\n'.join(preds))

        _style = os.path.join('./results', file.split('.'), '_style.txt')
        with open(_style, 'w') as g:
            g.write(style)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_test_imgs', help='path to the directory of test data', type=str)

    kwargs = vars(parser.parse_args())
    main(**kwargs)