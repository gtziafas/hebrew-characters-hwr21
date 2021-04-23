from typing import List, Tuple, Callable, TypeVar, Any, Dict
from typing import Optional as Maybe
from dataclasses import dataclass
from abc import ABC

from numpy import array 
from torch import Tensor
from torch import float as floatt
from torch import long as longt
from torch.utils.data import DataLoader 
from torch.optim import Optimizer 


CHARACTER_NAMES = {k: v for k, v in enumerate(['Alef',
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
 'Zayin'])}


@dataclass
class Character:
    image: array # H x W binary matrix
    label: int   # 0, 1, ..., 26
    name: str    # Alef, Ayin, ..., Zayin

    def __init__(self, image: array, label: Maybe[int] = None):
        self.image = image 
        self.label = label
        self.name = CHARACTER_NAMES[self.label]