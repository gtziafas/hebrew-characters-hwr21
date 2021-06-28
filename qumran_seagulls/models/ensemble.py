from ..types import *
from torch import no_grad, load, stack
from torch.nn import Module


class CNNEnsemble(object):
    def __init__(self, make_model: Callable[[], Module], checkpoint_path: str, device: str):
        self.model = make_model().eval().to(device)
        self.checkpoint = load(checkpoint_path) 
        self.device = device

    @no_grad()
    def predict_scores(self, inputs: Any) -> Tensor:
        all_scores = []
        for v in self.checkpoint:
            self.model.load_state_dict(v)
            all_scores.append(self.model.predict_scores(inputs, self.device))
        return stack(all_scores).mean(dim=0)