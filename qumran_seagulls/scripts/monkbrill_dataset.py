from .types import *
from .utils import show
from torchvision.datasets import ImageFolder


class MonkbrillDataset(ABC):
    def __init__(self, root: str):
        self.dataset = self.load_from_folders(root)
    
    def load_from_folders(self, root: str) -> List[Character]:
        dataset = ImageFolder(root=root)
        # return only first channel as images are binary anyway
        return [Character(image=array(x)[..., 0], label=y) for x, y in dataset]

    def __getitem__(self, n: int) -> Character:
        return self.dataset[n]

    def __len__(self) -> int:
        return len(self.dataset)

    def show(self, n: int):
        char = self.__getitem__(n)
        show(char.image, legend=char.name)