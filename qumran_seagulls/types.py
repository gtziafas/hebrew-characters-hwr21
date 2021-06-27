from typing import List, Tuple, TypeVar, Any, Dict, Set, Callable
from typing import Optional as Maybe
from typing import Mapping as Map
from dataclasses import dataclass
from abc import ABC

from numpy import array 
from torch import Tensor
from torch import float as floatt
from torch import long as longt
from torch.utils.data import DataLoader 
from torch.optim import Optimizer 


LABEL_MAP = {None: 'nan', **{k: v for k, v in enumerate([
 'Alef',
 'Ayin',
 'Bet',
 'Dalet',
 'Gimel',
 'He',
 'Het',
 'Kaf',
 'Kaf-final',
 'Lamed',
 'Mem',
 'Mem-medial',
 'Nun-final',
 'Nun-medial',
 'Pe',
 'Pe-final',
 'Qof',
 'Resh',
 'Samekh',
 'Shin',
 'Taw',
 'Tet',
 'Tsadi-final',
 'Tsadi-medial',
 'Waw',
 'Yod',
 'Zayin',
 'between'])}}
LABEL_MAP_INV = {v : k for k,v in LABEL_MAP.items()}

STYLE_MAP = {None: 'nan', 0: 'Archaic', 1: 'Hasmonean', 2: 'Herodian'}


@dataclass
class Character:
    image: array    # H x W binary matrix
    label: int      # 0, 1, ..., 26
    label_str: str  # Alef, Ayin, ..., Zayin
    style: int      # 0, 1, 2
    style_str: str  # Archaic, Hasmonean, Herodian

    def __init__(self, image: array, label: Maybe[int]=None, style: Maybe[int]=None):
        self.image = image 
        self.label = label
        self.label_str = LABEL_MAP[self.label]
        self.style = style 
        self.style_str = STYLE_MAP[self.style]


@dataclass 
class Box:
    # coordinates in (x,y) cv2-like frame
    x: int 
    y: int
    w: int 
    h: int 


class Pipeline(ABC):

    def preprocess(self, image: array) -> List[List[List[array]]]:
        # from full image -> lines x words x character images
        ...

    def predict_line(self, line: array) -> List[str]:
        ...

    def predict(self, lines: List[List[array]]) -> List[List[str]]:
        return list(map(predict_line, lines))
