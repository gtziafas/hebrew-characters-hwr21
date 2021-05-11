from ..types import *
from ..utils import crop_boxes_fixed

import cv2
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
from torch import tensor, stack, no_grad


class BaselineCNN(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 dropout_rates: List[float], 
                 inp_shape: Tuple[int, int]):
        super().__init__()
        self.inp_shape = inp_shape
        self.dropout_rates = dropout_rates
        self.block1 = self.block(in_channels=1, out_channels=16, conv_kernel=3, pool_kernel=3) 
        self.block2 = self.block(in_channels=16, out_channels=32, conv_kernel=3, pool_kernel=3)
        self.block3 = self.block(in_channels=32, out_channels=64, conv_kernel=3, pool_kernel=2)
        self.cls = nn.Linear(in_features=256, out_features=num_classes)

    def block(self, in_channels: int, out_channels: int, conv_kernel: int, pool_kernel: int, conv_stride: int=1):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, stride=conv_stride),
                             nn.GELU(),
                             nn.MaxPool2d(kernel_size=pool_kernel)
                            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = F.dropout(x, p = self.dropout_rates[0])
        x = self.block3(x)
        x = F.dropout(x, p = self.dropout_rates[1])
        x = x.flatten(1)
        return self.cls(x)

    @no_grad()
    def predict(self, imgs: List[array], device: str='cpu') -> str:
        padded = crop_boxes_fixed(self.inp_shape)(imgs)
        padded = [(img / 0xff).astype(np.float) for img in padded]
        tensorized = stack([tensor(img, dtype=floatt, device=device) for img in padded])
        predictions = self.forward(tensorized)
        return predictions.argmax(dim=-1)


def pad_imgs_in_batch(batch: List[array], desired_shape: Tuple[int, int]) -> List[array]:
    shapes = [img.shape for img in batch]

    # have fixed resolution frame
    center = (desired_shape[0] // 2, desired_shape[1] // 2)

    # paste every image in the center of the fixed resolution frame to pad all images
    offsets = [(center[0] - shape[0] // 2, center[1] - shape[1] // 2) for shape in shapes]
    batch_padded = []
    for img, shape, offset in zip(batch, shapes, offsets):
        batch_padded.append(np.zeros((desired_shape[0], desired_shape[1])))
        batch_padded[-1][offset[0] : offset[0] + shape[0], offset[1] : offset[1] + shape[1]] = img 
    return batch_padded


def collate(device: str, with_padding: Maybe[Tuple[int, int]]=None) -> Callable[[List[Character]], Tuple[Tensor, Tensor]]:
    
    def _collate(batch: List[Character]) -> Tuple[Tensor, Tensor]:
        imgs, labels = zip(*[(s.image, s.label) for s in batch])
    
        # pad to equal size if desired
        if with_padding is not None:
            imgs = pad_imgs_in_batch(imgs, desired_shape=with_padding)

        # normalize images to [0, 1] range
        imgs = [(img / 0xff).astype(np.float) for img in imgs]

        # tensorize, send to device, add channel dimension and stack
        imgs = stack([tensor(img, dtype=floatt, device=device) for img in imgs], dim=0).unsqueeze(1)
        labels = stack([tensor(label, dtype=longt, device=device) for label in labels], dim=0)
        return imgs, labels

    return _collate


def default_cnn():
    return BaselineCNN(num_classes=27, dropout_rates=[0, 0], inp_shape=(75, 75))