from ..types import *
from ..utils import show
from ..data.monkbrill_dataset import MonkbrillDataset 
import os


class StylesDataset(ABC):
    def __init__(self, 
                 root: str, 
                 with_preproc: Maybe[Callable[[List[array]], List[array]]] = None
                ):
        self.with_preproc = with_preproc
        self.dataset = self.load_from_folders(root)

    def load_from_folders(self, root: str) -> List[Character]:
        style_folders = next(os.walk(root))[1]
        style_datasets = [MonkbrillDataset(os.path.join(root, f), self.with_preproc) for f in style_folders]
        style_datasets = [[Character(image=s.image, label=s.label, style=i) for s in ds] for i, ds in enumerate(style_datasets)]
        return sum(style_datasets, [])

    def __getitem__(self, n: int) -> Character:
        return self.dataset[n]

    def __len__(self) -> int:
        return len(self.dataset)

    def show(self, n: int):
        char = self.__getitem__(n)
        show(char.image, legend='-'.join([char.label_str, char.style_str]))