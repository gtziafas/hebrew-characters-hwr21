from qumran_seagulls.models.cnn import default_cnn_monkbrill
from qumran_seagulls.utils import *
import torch

from math import ceil
import numpy as np
    
LOAD_PATH = 'checkpoints/cnn_labels_augm.p'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


def visualize(line: array, histogram: array, starts: List[float], ends: List[float]):
    from matplotlib import pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, sharex=True) 
    fig.suptitle('Character segmentation - adaptive pixel count') 
    ax1.imshow(line) 
    ax1.vlines(starts, 0, line.shape[0], color="C1") 
    ax1.vlines(ends, 0, line.shape[0], color="C2") 
    ax2.plot(histogram) 
    ax2.grid(True) 
    ax2.vlines(starts, 0, line.shape[0], color="C1") 
    ax2.vlines(ends, 0, line.shape[0], color="C2") 
    plt.show() 


def histogram_cleaning(line: array, width_thresh: int = 15, span_thresh: int = 10) -> List[array]:
    histogram = np.where(line > 0, 1, 0).sum(axis=0)
    zeros = np.where(histogram==0)[0]
    change_idces = np.where(np.diff(zeros) > 1)[0]
    starts = [zeros[i] for i in change_idces] 
    ends = [zeros[i+1] for i in change_idces]

    # remove very narrow zero areas as they might be of the same character
    widths = [e - s for s, e in zip(starts, ends)]
    remove = [i for i, w in enumerate(widths) if w < width_thresh]
    ends = [e for i, e in enumerate(ends) if i not in remove]
    starts = [s for i, s in enumerate(starts) if i not in [j+1 for j in remove]]

    # remove very narrow spans for the same reason
    spans = [starts[i+1] - e for i, e in enumerate(ends[:-1])]
    remove = [i for i, s in enumerate(spans) if s < span_thresh]
    ends = [e for i, e in enumerate(ends) if i not in remove]
    starts = [s for i, s in enumerate(starts) if i not in [j+1 for j in remove]]
    #visualize(line, histogram, starts, ends)

    return [line[:, s:e+16] for s, e in zip(starts, ends)]



def create_windows(image: array, win_width: int = 50, win_step: int = 15) -> List[array]:
    num_windows = ceil((image.shape[1] - win_width) / win_step)
    windows = [image[:, i*win_step : win_width + i*win_step] for i in range(num_windows)]
    return [w for w in windows if w.min() == 0]

    

class CharacterSegmenter(ABC):
    def __init__(self, load_path: str, device: str, width_thresh: int = 75):
        self.model = default_cnn_monkbrill()
        self.model.load_pretrained(load_path)
        self.device = device
        self.width_thresh = width_thresh


    def get_likelihoods(self, crop: array, scores: Tensor) -> Tensor:
        # heuristic
        num_chars = crop.shape[1] // 75 + 1
        max_probs = [score.softmax(-1).max() for score in scores]
        split_idces = sorted([max_probs.index(p) for p in sorted(max_probs, reverse=False)[:num_chars-1]])

        if num_chars == 1:
            lhds = [scores.mean(0).softmax(-1)]

        else:
            lhds = [scores[:split_idces[0]].mean(0)]
            lhds.extend([scores[split_idces[i] + 1 : idx].mean(0) for i, idx in enumerate(split_idces[1:-1])])
            lhds.append(scores[split_idces[-1]:].mean(0))
            lhds = [l.softmax(-1) for l in lhds if not torch.isnan(l[0])]

        return torch.stack(lhds)

    def __call__(self, line: array) -> List[Tensor]:
        line = remove_blobs(thresh_invert(line), area_thresh=10)
        crops = histogram_cleaning(line)
        all_windows = [create_windows(w) for w in crops]
        crops = [cs for i, cs in enumerate(crops) if len(all_windows[i]) > 0]
        all_windows = [ws for ws in all_windows if len(ws) > 0]

        all_scores = [self.model.predict_scores(wins) for wins in all_windows]

        for crop, windows, scores in zip(crops, all_windows, all_scores):
            print(self.get_likelihoods(crop, scores))

        return [self.get_likelihoods(crop, scores) for crop, scores in zip(crops, all_scores)]
    
    def debug(self, line: array) -> List[List[array]]:
        line = remove_blobs(thresh_invert(line), area_thresh=10)
        crops = histogram_cleaning(line)
        all_windows = [create_windows(w) for w in crops]
        all_windows = [ws for ws in all_windows if len(ws) > 0]

        all_scores = [self.model.predict_scores(wins) for wins in all_windows]

        cv2.imshow('line', line)
        for crop, windows, scores in zip(crops, all_windows, all_scores):
            print(f'New crop... {crop.shape}')
            max_probs = []; projections = []
            cv2.imshow('crop', crop)
            print(self.get_likelihoods(crop, scores))
            # histogram = np.where(crop>0, 1, 0).sum(axis=0)
            for i, win in enumerate(windows):
                print([LABEL_MAP[s.item()] for s in scores[i].topk(k=3).indices], [s for s in scores[i].softmax(-1).topk(k=3).values])
                # max_probs.append(scores[i].softmax(-1).max())
                # projections.append(histogram[i*10 : 50 + i*10].sum())
                show(win)
            # from matplotlib import pyplot as plt
            # fig, (ax1, ax2) = plt.subplots(2, sharex=False) 
            # ax1.imshow(crop)
            # ax2.plot(max_probs)
            # ax2.hlines(min(0.8, max(max_probs)-0.1), 0, len(max_probs), color="C1")
            # ax2.grid(True)
            # # ax2.plot([p/max(projections) for p in projections])
            # plt.show() 
    