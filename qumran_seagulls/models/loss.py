from ..types import *

import torch.nn as nn
from torch import tensor, zeros_like
from math import factorial


# n-th order approximation of softmax with taylor series, aimed to reduce inter-class distance
class TaylorSoftmax(nn.Module):
    def __init__(self, order: int):
        super().__init__()
        self.coeffs = [1 / factorial(n) for n in range(order + 1)]  # N+1,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        self.powers = list(range(order + 1))     # N+1,

    def forward(self, x: Tensor) -> Tensor:
        batch_size, vec_size = x.shape      # B, D
        coeffs = tensor(self.coeffs, dtype=floatt, device=x.device).repeat(batch_size, vec_size, 1)  # B x D x N+1
        powers = tensor(self.powers, dtype=longt, device=x.device).repeat(batch_size, vec_size, 1)   # B x D x N+1
        taylor = x.unsqueeze(2).pow(powers).mul(coeffs).sum(dim=-1)   # B x D
        norm = taylor.sum(dim=-1).unsqueeze(1)    # B x 1
        return taylor.div(norm)        # B x D
                            

# introduce soft-margin hyperparam inside softmax, aimed to reduce intra-class distance 
class SoftMarginSoftmax(nn.Module):
    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, x: Tensor) -> Tensor:
        ex, ex_m = x.exp(), x.subtract(self.margin).exp()               # B x D, B x D
        norm = ex.sum(dim=-1).unsqueeze(1).repeat(1, x.shape[-1]) - ex  # B x D
        return ex_m.div(ex_m + norm)                                    # B x D


class SoftMarginTaylorSoftmax(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass


class FuzzyLoss(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 mass_redistribution: float, 
                 softmax: nn.Module = nn.Softmax(dim=-1),
                 reduction: str = 'batchmean'
                ):
        super().__init__()
        self.num_classes = num_classes
        self.mass_redistribution = mass_redistribution
        self.softmax = softmax
        self.KLD = nn.KLDivLoss(reduction=reduction)

    def forward(self, predictions: Tensor, truth: Tensor) -> Tensor:
        fuzzy = zeros_like(predictions, dtype=floatt, device=predictions.device)
        fuzzy.fill_(self.mass_redistribution / (self.num_classes - 1))
        fuzzy.scatter_(1, truth.unsqueeze(1), 1 - self.mass_redistribution)
        log_probs = self.softmax(predictions).log()
        return self.KLD(log_probs, fuzzy)