import matplotlib.pyplot as plt
import numpy as np

from qumran_seagulls.types import *
from qumran_seagulls.models.cnn import concat_cnn_styles, default_cnn_monkbrill, default_cnn_styles


# Accuracies by running style classification separately per character.
from qumran_seagulls.utils import resize

VOTE_WEIGHTS = [0.510, 0.700, 0.555, 0.740, 0.727, 0.574, 0.623, 0.690, 0.630, 0.537, 0.550, 0.710, 0.747, 0.577, 0.631,
                0.710, 0.547, 0.623, 0.648, 0.782, 0.700, 0.720, 0.705, 0.760, 0.840, 0.720, 0.700]


class StyleClassifier(ABC):
    def __init__(self, load_path: str, device: str, vote_weights: List[float] = VOTE_WEIGHTS,
                 char_label_cnn_load_path: str = 'data/saved_models/cnn_labels.p'):
        self.vote_weights = vote_weights
        self.cnn = default_cnn_styles().eval().to(device)
        self.cnn.load_pretrained(load_path)
        self.char_label_cnn = default_cnn_monkbrill().eval().to(device)
        self.char_label_cnn.load_pretrained(char_label_cnn_load_path)
        self.device = device
        self.out_map = {0: 'Archaic', 1: 'Hasmonean', 2: 'Herodian'}

    def predict(self, chars: List[array], debug=False) -> str:
        char_labels = self.char_label_cnn.predict_scores(chars, self.device).argmax(-1).cpu().tolist()

        # get classification scores for each character
        scores = self.cnn.predict_scores(chars, self.device).cpu()

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
