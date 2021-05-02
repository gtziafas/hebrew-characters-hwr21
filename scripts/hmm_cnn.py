from .types import *

import torch.nn as nn
import torch
from opt_einsum import contract


class HMMCNN(Model):
    def __init__(self, 
                 cnn: nn.Module, 
                 num_classes: int, 
                 transition_matrix: Tensor,
                 transition_start: Tensor,
                 transition_end: Tensor):
        super().__init__()
        self.cnn = cnn 
        self.num_classes = num_classes
        self.T = transition_matrix      # K x K
        self.T_sos = transition_start   # K,
        self.T_eos = transition_end     # K,

    def preprocess(self, image: array) -> List[Line]:
        ...

    @torch.no_grad()
    def predict_line(self, line: Line) -> List[str]:
        T = len(line) # sequence length
        K = self.num_classes

        cnn_input = self.tensorize_line(line)           #T x K
        preds = self.cnn(cnn_input).softmax(dim=-1)     #T x K

        qs = np.empty((T+1, K))          # T+1 x K
        qs[0, :] = preds[0, :] * self.T_sos     # K
        for step in range(2, T):
            # @todo: find the broadcasting magic
            q[step] = preds[step-1] * (q[step-1] * T[...]).max()  
        q[step+1, :] = (q[step-1, :] * self.T_eos).max()


        best_path = self.viterbi(q)
        return list(map(LABEL_MAP, best_path))


    def viterbi(self, Q: Tensor) -> List[int]:
        return Q.argmax(dim=-1).cpu().tolist() # T x K -> T,

    def tensorize_line(self, line: Line) -> Tensor:
        ...