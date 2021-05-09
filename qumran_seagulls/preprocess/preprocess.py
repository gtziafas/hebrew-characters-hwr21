from ..types import *

LineModule = Callable[[array], List[array]]
CharModule = Callable[[array], List[List[array]]]


class Preprocess(ABC):
    def __init__(self,
                 line_module:, 
                 line_cfg:,
                 char_module:,
                 char_cfg:, 
                 )
        self.line_cfg = 
        self.line_module = line_module(**self.line_cfg)
        self.char_cfg = 
        self.char_module = char_module(**self.char_cfg)

    def segment_lines(image: array) -> List[array]:
        return self.line_module(image)

    def segment_characters(line: array) -> List[List[array]]:
        return self.char_module(line)

    def __call__(image: array) -> List[List[List[array]]]:
        lines = self.segment_lines(image)
        characters = list(map(self.segment_characters, lines))
        return characters