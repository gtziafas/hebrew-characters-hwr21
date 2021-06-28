from qumran_seagulls.types import *
from qumran_seagulls.utils import thresh_invert_many
from qumran_seagulls.models.cnn import default_cnn_monkbrill, default_cnn_styles
from qumran_seagulls.preprocess.char_segm.char_segm_astar import segment_characters

import matplotlib.pyplot as plt
import numpy as np

VOTE_WEIGHTS = [0.510, 0.700, 0.555, 0.740, 0.727, 0.574, 0.623, 0.690, 0.630, 0.537, 0.550, 0.710, 0.747, 0.577, 0.631,
                0.710, 0.547, 0.623, 0.648, 0.782, 0.700, 0.720, 0.705, 0.760, 0.840, 0.720, 0.700]


class StyleClassifier(ABC):
    def __init__(self, styles_load_path: str, labels_load_path: str, device: str, vote_weights: List[float] = VOTE_WEIGHTS):
        self.vote_weights = vote_weights
        self.cnn_styles = default_cnn_styles().eval().to(device)
        self.cnn_styles.load_pretrained(styles_load_path)
        self.cnn_labels = default_cnn_monkbrill().eval().to(device)
        self.cnn_labels.load_pretrained(labels_load_path)
        self.device = device
        self.out_map = {0: 'Archaic', 1: 'Hasmonean', 2: 'Herodian'}

    def predict(self, chars: List[array], debug: bool =False) -> str:
        char_labels = self.cnn_labels.predict_scores(chars, self.device).argmax(-1).cpu().tolist()
        if debug:
            print(f"char labels: {char_labels}")

        # get classification scores for each character
        scores = self.cnn_styles.predict_scores(chars, self.device).cpu()

        # get predictions for style and their corresponding vote weights
        predictions = np.argmax(scores, axis=-1)

        votes = [score * self.vote_weights[char_label] for score, char_label in zip(scores, char_labels)]

        arch_votes = sum([w[0] for w, p in zip(votes, predictions) if p == 0])
        hasm_votes = sum([w[1] for w, p in zip(votes, predictions) if p == 1])
        herod_votes = sum([w[2] for w, p in zip(votes, predictions) if p == 2])

        if debug:
            print((arch_votes, hasm_votes, herod_votes))
            print(np.argmax([arch_votes, hasm_votes, herod_votes], axis=-1))

        return self.out_map[np.argmax([arch_votes, hasm_votes, herod_votes], axis=-1)]

    def __call__(self, lines: List[array], debug: bool = False) -> str:
        lines = [l/0xff for l in thresh_invert_many(lines)]
        all_chars = sum(list(map(segment_characters, lines)), [])
        return self.predict([(c * 0xff).astype("uint8") for c in all_chars], debug=debug)


def default_style_classifier(device: str):
    return StyleClassifier('checkpoints/cnn_styles.p', 'checkpoints/cnn_labels.p', device=device)