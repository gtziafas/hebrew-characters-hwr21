from ..types import *
from ..utils import show, filter_large

from torchvision.datasets import ImageFolder


class MonkbrillDataset(ABC):
    def __init__(self, 
                 root: str, 
                 with_preproc: Maybe[Callable[[List[array]], List[array]]] = None
                ):
        self.with_preproc = with_preproc
        self.dataset = self.load_from_folders(root)
    
    def load_from_folders(self, root: str) -> List[Character]:
        dataset = ImageFolder(root=root)
        # return only first channel as images are binary anyway
        imgs, labels = zip(*[(array(x)[..., 0], y) for x, y in dataset])
        # apply desired preprocessing if given
        imgs = self.with_preproc(imgs) if self.with_preproc is not None else imgs
        return [Character(image=x, label=y) for x, y in zip(imgs, labels)]

    def __getitem__(self, n: int) -> Character:
        return self.dataset[n]

    def __len__(self) -> int:
        return len(self.dataset)

    def show(self, n: int):
        char = self.__getitem__(n)
        show(char.image, legend=char.label_str)


def get_monkbrill():
    return MonkbrillDataset('./data/monkbrill', with_preproc=filter_large((75, 75)))