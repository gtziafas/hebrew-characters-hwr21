from ..types import *
from ..utils import *

import torch.nn as nn 
from torch import tensor, stack, no_grad, load, cat, zeros


class CNNFeatures(nn.Module):
    def __init__(self, num_blocks: int, dropout_rates: List[float], 
                 conv_kernels: List[int], pool_kernels: List[int],
                 input_channels: int = 1):
        super().__init__()
        assert num_blocks == len(conv_kernels) == len(pool_kernels) == len(dropout_rates)
        self.blocks = nn.Sequential(self.block(input_channels, 16, conv_kernels[0], pool_kernels[0], dropout_rates[0]),
                                    *[self.block(2**(3+i), 2**(4+i), conv_kernels[i], pool_kernels[i], dropout_rates[i])
                                    for i in range(1, num_blocks)])

    def block(self, 
              in_channels: int, 
              out_channels: int, 
              conv_kernel: int, 
              pool_kernel: int, 
              dropout: float = 0., 
              conv_stride: int = 1
             ):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, stride=conv_stride),
                             nn.GELU(),
                             nn.MaxPool2d(kernel_size=pool_kernel),
                             nn.Dropout(p=dropout)
                            ) 

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x).flatten(1) # B x D


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int, dropout_rates: List[float], inp_shape: Tuple[int, int], num_features: int,
                 with_preproc: Maybe[Callable[[List[array]], List[array]]] = None,
                 conv_kernels: List[int] = [3, 3, 3], pool_kernels: List[int] = [3, 2, 2]):
        super().__init__()
        self.with_preproc = with_preproc
        self.inp_shape = inp_shape
        self.features = CNNFeatures(3, dropout_rates, conv_kernels, pool_kernels)
        self.cls = nn.Linear(in_features=num_features, out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: B x 1 x H x W
        x = self.features(x)
        return self.cls(x)

    @no_grad()
    def predict_scores(self, imgs: List[array], device: str='cpu') -> Tensor:
        self.eval()
        # filtered = filter_large(self.inp_shape)(imgs)
        # padded = pad_with_frame(filtered, self.inp_shape)
        imgs = self.with_preproc(list(imgs)) if self.with_preproc is not None else imgs
        tensorized = stack([tensor(img / 0xff, dtype=floatt, device=device) for img in imgs])
        scores = self.forward(tensorized.unsqueeze(1))
        return scores

    @no_grad()
    def predict(self, imgs: List[array], device: str='cpu') -> List[str]:
        predictions = self.predict_scores(imgs, device).argmax(-1).cpu()
        return [LABEL_MAP[label] for label in predictions]

    def load_pretrained(self, path: str):
        checkpoint = load(path)
        self.load_state_dict(checkpoint)


class ConcatCNN(nn.Module):
    def __init__(self, num_classes: int, dropout_rates: List[float], inp_shape: Tuple[int, int], num_features: int,
                 with_preproc: Maybe[Callable[[List[array]], List[array]]] = None, num_char_labels: int = 27,
                 conv_kernels: List[int] = [3, 3, 3], pool_kernels: List[int] = [3, 2, 2]):
        super().__init__()
        self.num_char_labels = num_char_labels
        self.with_preproc = with_preproc
        self.inp_shape = inp_shape
        self.features = CNNFeatures(3, dropout_rates, conv_kernels, pool_kernels)
        self.cls = nn.Linear(in_features=num_features + num_char_labels, out_features=num_classes)

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tensor:
        # x: B x 1 x H x W,     y: B x K
        x, y = inputs
        x = self.features(x)
        y_onehot = zeros(y.shape[0], self.num_char_labels, device=x.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
        x = cat((x, y_onehot), dim=-1)
        return self.cls(x)

    @no_grad()
    def predict_scores(self, imgs: List[array], labels: List[int], device: str='cpu') -> Tensor:
        self.eval()
        # filtered = filter_large(self.inp_shape)(imgs)
        # padded = pad_with_frame(filtered, self.inp_shape)
        imgs = self.with_preproc(list(imgs)) if self.with_preproc is not None else imgs
        tensorized = stack([tensor(img / 0xff, dtype=floatt, device=device) for img in imgs])
        labels = stack([tensor(y, dtype=longt, device=device) for y in labels])
        scores = self.forward((tensorized.unsqueeze(1), labels))
        return scores

    @no_grad()
    def predict(self, imgs: List[array], labels: List[int], device: str='cpu') -> List[str]:
        predictions = self.predict_scores(imgs, labels, device).argmax(-1).cpu()
        return [LABEL_MAP[label] for label in predictions]

    def load_pretrained(self, path: str):
        checkpoint = load(path)
        self.load_state_dict(checkpoint)


class MultiTaskCNN(nn.Module):
    def __init__(self, num_style_labels: int, dropout_rates: List[float], inp_shape: Tuple[int, int], num_features: int,
                 with_preproc: Maybe[Callable[[List[array]], List[array]]] = None, num_char_labels: int = 27,
                 conv_kernels: List[int] = [3, 3, 3], pool_kernels: List[int] = [3, 2, 2]):
        super().__init__()
        self.num_char_labels = num_char_labels
        self.with_preproc = with_preproc
        self.inp_shape = inp_shape
        self.features = CNNFeatures(3, dropout_rates, conv_kernels, pool_kernels)
        self.label_cls = nn.Linear(in_features=num_features, out_features=num_char_labels)
        self.style_cls = nn.Linear(in_features=num_features, out_features=num_style_labels)

    def forward(self, x: Tensor) -> Tensor:
        # x: B x 1 x H x W
        x = self.features(x)
        return self.label_cls(x), self.style_cls(x)

    @no_grad()
    def predict_scores(self, imgs: List[array], device: str='cpu') -> Tuple[Tensor, Tensor]:
        self.eval()
        imgs = self.with_preproc(list(imgs)) if self.with_preproc is not None else imgs
        tensorized = stack([tensor(img / 0xff, dtype=floatt, device=device) for img in imgs])
        return self.forward(tensorized.unsqueeze(1))
        
    @no_grad()
    def predict(self, imgs: List[array], device: str='cpu') -> List[Tuple[str, str]]:
        label_preds, style_preds = self.predict_scores(imgs, labels, device)
        label_preds = label_preds.argmax(-1).cpu().tolist()
        style_preds = style_preds.argmax(-1).cpu().tolist()
        return [(LABEL_MAP[label], STYLE_MAP[style]) for label, styles in zip(label_preds, style_preds)]

    def load_pretrained(self, path: str):
        checkpoint = load(path)
        self.load_state_dict(checkpoint)

    def load_char_model(self, path: str):
        checkpoint = load(path)
        self.features.load_state_dict(checkpoint['features'])
        self.label_cls.load_state_dict(checkpoint['cls'])


def collate(device: str, with_padding: Maybe[Tuple[int, int]] = None) -> Callable[[List[Character]], Tuple[Tensor, Tensor]]:
    
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


def collate_concat(device: str, with_padding: Maybe[Tuple[int, int]] = None
    ) -> Callable[[List[Character]], Tuple[Tuple[Tensor, Tensor], Tensor]]:
    
    def _collate(batch: List[Character]) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        imgs, labels, styles = zip(*[(s.image, s.label, s.style) for s in batch])
        if with_padding is not None:
            imgs = pad_with_frame(imgs, desired_shape=with_padding)
        imgs = stack([tensor(img / 0xff, dtype=floatt, device=device) for img in imgs]).unsqueeze(1)
        labels = stack([tensor(label, dtype=longt, device=device) for label in labels])
        styles = stack([tensor(style, dtype=longt, device=device) for style in styles])
        return (imgs, labels), styles

    return _collate


def default_cnn_monkbrill() -> BaselineCNN:
    return BaselineCNN(num_classes=27, dropout_rates=[0., 0.1, 0.5], inp_shape=(75, 75), num_features=1024,
                       with_preproc=center_of_gravities((75, 75)))


def monkbrill_with_between_class() -> BaselineCNN:
    return BaselineCNN(num_classes=28, dropout_rates=[0., 0.1, 0.5], inp_shape=(75, 75), num_features=1024,
                       with_preproc=resize((75, 75)))


def default_cnn_styles() -> BaselineCNN:
    return BaselineCNN(num_classes=3, dropout_rates=[0., 0.1, 0.5], inp_shape=(75, 75), num_features=1024,
                       with_preproc=crop_boxes_fixed((75, 75)))

def concat_cnn_styles() -> ConcatCNN:
    return ConcatCNN(num_classes=3, dropout_rates=[0., 0.1, 0.5], inp_shape=(75, 75), num_features=1024,
                       with_preproc=crop_boxes_fixed((75, 75)), num_char_labels=27)

def multitask_cnn() -> MultiTaskCNN:
    return MultiTaskCNN(num_style_labels=3, dropout_rates=[0., 0.1, 0.75], inp_shape=(75, 75), num_features=1024,
                       with_preproc=crop_boxes_fixed((75, 75)), num_char_labels=27)