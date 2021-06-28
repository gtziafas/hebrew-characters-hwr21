import cv2
import matplotlib.pyplot as plt

from qumran_seagulls.models.style_classifier import StyleClassifier
from qumran_seagulls.preprocess.char_segm.char_segm_astar import segment_characters
from qumran_seagulls.preprocess.line_segm.line_segm_astar import call_lineSeg

from qumran_seagulls.types import *


def classify_lines_style(lines: List[array], debug=False):
    style_classifier = StyleClassifier(load_path="data/saved_models/cnn_styles.p", device="cpu")
    chars_in_img = []
    for line in lines:
        segm_chars = segment_characters(line)
        chars_in_img += segm_chars

    prediction = style_classifier.predict([(c).astype("uint8") for c in chars_in_img], debug=debug)
    return prediction


if __name__ == '__main__':
    file_id = "P106-Fg002-R-C01-R01"

    line_imgs = []
    for i in range(1, 4):
        line_imgs.append((0xff - cv2.imread(str(f"data/lines_cropped/{file_id}/line_{i}.jpg"), cv2.IMREAD_GRAYSCALE)) / 0xff)
    print(f"imgs shapes {[l.shape for l in line_imgs]}")

    print(classify_lines_style(line_imgs, debug=True))

# command:
# python3 -m qumran_seagulls.scripts.classify_lines_style
# from project root folder
