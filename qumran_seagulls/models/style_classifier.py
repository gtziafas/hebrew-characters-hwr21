from qumran_seagulls.types import *
from qumran_seagulls.models.cnn import concat_cnn_styles
from qumran_seagulls.preprocess.char_segm import create_windows

VOTE_WEIGHTS = [0, 0, 0]

class StyleClassifier(ABC):
    def __init__(self, load_path: str, device: str, vote_weights: List[float] = VOTE_WEIGHTS):
        self.vote_weights = vote_weights
        self.cnn = concat_cnn_styles().eval().to(device)
        self.cnn.load_pretrained(load_path)
        self.device = device
        self.out_map = {0: 'Archaic', 1: 'Hasmonean', 2: 'Herodian'}

    def predict(self, chars: List[array]) -> str:
        # get classification scores for each character
        scores = self.cnn.predict_scores(imgs, self.device).cpu().tolist()

        # get predictions for style and their corresponding vote weights
        predictions = scores.argmax(-1).tolist()
        votes = [score * self.vote_weights[pred] for score, pred in zip(scores, predictions)]

        arch_votes = sum([w for w, p in zip(votes, predictions) if p == 0])
        hasm_votes = sum([w for w, p in zip(votes, predictions) if p == 1])
        herod_votes = sum([w for w, p in zip(votes, predictions) if p == 2])
        
        return self.out_map[array(arch_votes, hasm_votes, herod_votes).argmax(-1)]

    def __call__(self, crops: List[array], width_thresh: int = 75):
        all_windows = [create_windows(w) for w in crops]
        all_windows = [ws for ws in all_windows if len(ws) > 0]
        all_scores = [self.cnn.predict_scores(wins) for wins in all_windows]

        for crop, scores in zip(crops, all_scores):
            max_probs = [score.softmax(-1).max() for score in scores]
            # heuristic
            num_chars = crop.shape[1] // width_thresh + 1
            return_windows = []
            for i in range(num_chars):
                max_probs[i * width_thresh: (i+1) * width_thresh].argmax()
        self.cnn.predict_scores()