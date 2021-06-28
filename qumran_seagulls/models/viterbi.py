from ..types import *

import torch.nn as nn
import torch


class Viterbi(object):
    def __init__(self, 
                 num_classes: int, 
                 transition_matrix: Tensor):
        super().__init__()
        self.num_classes = num_classes
        self.transition = transition_matrix      # K x K
        # remove transition probs from corresponding medial and final characters
        self.T_sos = torch.tensor([1/23 if i not in [8, 12, 15, 22] else 0 for i in range(num_classes)])
        self.T_eos = torch.tensor([1/24 if i not in [11, 13, 23] else 0 for i in range(num_classes)])
        

    @torch.no_grad()
    def __call__(self, likelihoods: Tensor) -> Tuple[List[int], Tensor, float]:
        T = len(likelihoods) # sequence length
        K = self.num_classes

        qs = torch.empty((T, K))         
        paths = torch.empty((T-1, K), dtype=int)
        qs[0, :] = likelihoods[0, :] * self.T_sos     
        for step in range(1, T):
            # @todo: find the broadcasting magic
            for j in range(K):
                query = (qs[step-1,:] * self.transition.T[j] * likelihoods[step, j])
                paths[step-1, j] = query.argmax()
                qs[step, j] = query.max()
        last_path = (qs[step, :] * self.T_eos).argmax().item()
        qs_end = (qs[step, :] * self.T_eos).max().item()

        best_path = [last_path]
        for step in range(T-1):
            last_path = paths[-1 - step, last_path].item()
            best_path.append(last_path)
            
        return best_path, qs, qs_end

