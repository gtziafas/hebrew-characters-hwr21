from ..types import *
from ..utils import pad_with_frame, filter_large

import torch.nn as nn 
from torch import tensor, stack, no_grad, load


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int, dropout_rates: List[float], inp_shape: Tuple[int, int], num_features: int):
        super().__init__()
        self.inp_shape = inp_shape
        self.block1 = self.block(in_channels=1, out_channels=16, conv_kernel=3, pool_kernel=3, dropout=0.) 
        self.block2 = self.block(in_channels=16, out_channels=32, conv_kernel=3, pool_kernel=2, dropout=dropout_rates[0])
        self.block3 = self.block(in_channels=32, out_channels=64, conv_kernel=3, pool_kernel=2, dropout=dropout_rates[1])
        self.cls = nn.Linear(in_features=num_features, out_features=num_classes)

    def block(self, 
              in_channels: int, 
              out_channels: int, 
              conv_kernel: int, 
              pool_kernel: int, 
              dropout: float, 
              conv_stride: int=1
             ):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, stride=conv_stride),
                             nn.GELU(),
                             nn.MaxPool2d(kernel_size=pool_kernel),
                             nn.Dropout(p=dropout)
                            )

    def forward(self, x: Tensor) -> Tensor:
        # x: B x 1 x H x W
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)
        return self.cls(x)

    @no_grad()
    def predict_scores(self, imgs: List[array], device: str='cpu') -> Tensor:
        self.eval()
        filtered = filter_large(self.inp_shape)(imgs)
        padded = pad_with_frame(filtered, self.inp_shape)
        tensorized = stack([tensor(img / 0xff, dtype=floatt, device=device) for img in padded])
        scores = self.forward(tensorized.unsqueeze(1))
        return scores

    @no_grad()
    def predict(self, imgs: List[array], device: str='cpu') -> List[str]:
        predictions = self.predict_scores(imgs, device).argmax(-1).cpu()
        return [LABEL_MAP[label] for label in predictions]

    def load_pretrained(self, path: str):
        checkpoint = load(path)
        self.load_state_dict(checkpoint)


def collate(device: str, with_padding: Maybe[Tuple[int, int]]=None) -> Callable[[List[Character]], Tuple[Tensor, Tensor]]:
    
    def _collate(batch: List[Character]) -> Tuple[Tensor, Tensor]:
        imgs, labels = zip(*[(s.image, s.label) for s in batch])
    
        # pad to equal size if desired
        if with_padding is not None:
            imgs = pad_with_frame(imgs, desired_shape=with_padding)

        # normalize to [0,1], tensorize, send to device, add channel dimension and stack
        imgs = stack([tensor(img / 0xff, dtype=floatt, device=device) for img in imgs], dim=0).unsqueeze(1)
        labels = stack([tensor(label, dtype=longt, device=device) for label in labels], dim=0)
        return imgs, labels

    return _collate


def default_cnn_monkbrill(dropout: Maybe[List[float]] = None) -> BaselineCNN:
    if dropout is None or dropout[0] is None or dropout[1] is None:
        dropout = [0.1, 0.5]
    return BaselineCNN(num_classes=27, dropout_rates=dropout, inp_shape=(75, 75), num_features=1024)


def default_cnn_styles() -> BaselineCNN:
    return BaselineCNN(num_classes=3, dropout_rates=[0.1, 0.5], inp_shape=(75, 75), num_features=1024)