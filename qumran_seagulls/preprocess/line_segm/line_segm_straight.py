import os
from math import sqrt

import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np

from qumran_seagulls.preprocess.shared_astar_funcs.persistence1d import RunPersistence

debug = True
min_persistence = 110

i = 0

def crop_straight_from_minima(image, minima):
    h, w = np.shape(image)
    l = len(minima)
    for m in range(l - 1):
        cropped_img = image[minima[m]:minima[m + 1], 0:w]
        if debug:
            plt.imshow(cropped_img)
            plt.show()


def segment_img(image):
    global i
    h, w = np.shape(image)

    blurred = cv2.blur(image, (15,15))
    histogram = numpy.sum(blurred, axis=1)
    print(f"max {np.max(histogram)}")
    if debug:
        plt.figure(i)
        i+=1
        plt.imshow(image)
        plt.plot(histogram, range(len(histogram)))

    max_histogram_value = np.max(histogram)
    scaled_min_persistence = min_persistence * sqrt(max_histogram_value) / 32

    extrema = RunPersistence(histogram)
    minima = extrema[0::2]  # odd elements are minima
    filtered_minima = [t[0] for t in minima if t[1] > scaled_min_persistence and histogram[t[0]] < 0.5 * max_histogram_value]
    sorted_minima = sorted(filtered_minima)
    print(len(sorted_minima))
    print(sorted_minima)

    if debug:
        plt.plot(histogram[sorted_minima], sorted_minima, "x")
        plt.hlines(y=sorted_minima, xmin=0, xmax=w, color="green")
        plt.show()

    # crop_straight_from_minima(image, sorted_minima)


def main():
    binarized_filenames = [x for x in os.listdir("data/images") if x.endswith("binarized.jpg")]

    for b in binarized_filenames:
        example_img_path = f"data/images/{b}"
        print(example_img_path)
        example_img = (255 - cv2.imread(str(example_img_path), cv2.IMREAD_GRAYSCALE)) / 255
        print(f"sum {np.sum(example_img)}")
        segment_img(example_img)


if __name__ == "__main__":
    main()

# command:
# python3 -m qumran_seagulls.preprocess.line_segm.line_segm_straight
# from project root folder
