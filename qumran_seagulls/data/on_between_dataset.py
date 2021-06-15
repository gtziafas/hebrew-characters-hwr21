import random

import numpy as np

from ..types import *
from ..utils import show, filter_large

from torchvision.datasets import ImageFolder


def pad_white_centered(img, target_h):
    """
    Pad with white to bring the image to some target height, while keeping it in the center
    :param img: original image
    :param target_h: target height
    :return: padded image
    """
    h, w = np.shape(img)
    top_padding = np.ones((int((target_h - h) / 2), w)) * 0xff
    img = np.vstack([top_padding, img])

    bottom_padding = np.ones((target_h - h - top_padding.shape[0], w)) * 0xff
    img = np.vstack([img, bottom_padding])

    return img


def make_between_img(im1: np.ndarray, im2: np.ndarray):
    h1, w1 = np.shape(im1)  # assuming both images have same size
    h2, w2 = np.shape(im2)  # assuming both images have same size

    right_im1 = im1[:, :int(w1/2)]
    left_im2 = im2[:, int(w2/2):]


    if h1 > h2:
        left_im2 = pad_white_centered(left_im2, h1)
    elif h2 > h1:
        right_im1 = pad_white_centered(right_im1, h2)

    return np.hstack([right_im1, left_im2])


class OnBetweenDataset(ABC):
    def __init__(self,
                 root: str,
                 with_preproc: Maybe[Callable[[List[array]], List[array]]] = None
                 ):
        self.with_preproc = with_preproc
        self.dataset = self.load_from_folders(root)

    def load_from_folders(self, root: str) -> List[Character]:
        on_dataset = ImageFolder(root=root)
        # return only first channel as images are binary anyway
        # discard individual labels, as they don't matter for segmentation
        imgs, labels = zip(*[(array(x)[..., 0], 0) for x, _ in on_dataset])

        between_imgs = []
        between_data_size = len(on_dataset)
        for _ in range(between_data_size):
            im1 = random.choice(imgs)
            im2 = random.choice(imgs)
            between_imgs.append((make_between_img(im1, im2), "between"))

        print(len(imgs))
        print(len(imgs[0]))
        print(type(imgs))
        print(type(imgs[0]))
        # add the "between" images, with label 1
        # imgs = imgs + tuple(between_imgs)
        # labels = labels + tuple([1] * len(between_imgs))

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
    return OnBetweenDataset('./data/monkbrill', with_preproc=filter_large((75, 75)))

get_monkbrill()