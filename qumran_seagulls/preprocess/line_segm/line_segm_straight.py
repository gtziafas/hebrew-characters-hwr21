import os

import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np

from qumran_seagulls.preprocess.shared_astar_funcs.persistence1d import RunPersistence

debug = True
min_persistence = 250


def crop_straight_from_minima(image, minima):
    h, w = np.shape(image)
    l = len(minima)
    for m in range(l - 1):
        cropped_img = image[minima[m]:minima[m + 1], 0:w]
        if debug:
            plt.imshow(cropped_img)
            plt.show()


def segment_img(image):
    h, w = np.shape(image)

    histogram = numpy.sum(image, axis=1)
    print(f"max {np.max(histogram)}")
    if debug:
        plt.imshow(image)
        plt.plot(histogram, range(len(histogram)))

    max_histogram_value = np.max(histogram)
    scaled_min_persistence = min_persistence * max_histogram_value / 1000

    extrema = RunPersistence(histogram)
    minima = extrema[0::2]  # odd elements are minima
    filtered_minima = [t[0] for t in minima if t[1] > scaled_min_persistence and histogram[t[0]] < 0.5 * max_histogram_value]
    sorted_minima = sorted(filtered_minima)
    print(len(sorted_minima))
    print(sorted_minima)

    plt.plot(histogram[sorted_minima], sorted_minima, "x")
    plt.hlines(y=sorted_minima, xmin=0, xmax=w, color="green")

    if debug:
        plt.show()

    # crop_straight_from_minima(image, sorted_minima)


def main():
    binarized_filenames = [x for x in os.listdir("data/images") if x.endswith("binarized.jpg")]

    for b in binarized_filenames:
        example_img_path = f"data/images/{b}"
        example_img = (255 - cv2.imread(str(example_img_path), cv2.IMREAD_GRAYSCALE)) / 255
        print(f"sum {np.sum(example_img)}")
        segment_img(example_img)


if __name__ == "__main__":
    main()

# command:
# python3 -m qumran_seagulls.preprocess.line_segm.line_segm_straight
# from project root folder
