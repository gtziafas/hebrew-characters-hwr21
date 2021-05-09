import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np

from persistence1d import RunPersistence

debug = True
min_persistence = 150


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
    if debug:
        plt.imshow(image)
        plt.plot(histogram, range(len(histogram)))

    extrema = RunPersistence(histogram)
    minima = extrema[0::2]  # odd elements are minima
    filtered_minima = [t[0] for t in minima if t[1] > min_persistence]
    sorted_minima = sorted(filtered_minima)
    print(sorted_minima)

    plt.plot(histogram[sorted_minima], sorted_minima, "x")
    plt.hlines(y=sorted_minima, xmin=0, xmax=w, color="green")

    if debug:
        plt.show()

    crop_straight_from_minima(image, sorted_minima)


def main():
    example_img_path = "../data/images/P123-Fg001-R-C01-R01-binarized.jpg"
    example_img = (255 - cv2.imread(str(example_img_path), cv2.IMREAD_GRAYSCALE)) / 255
    segment_img(example_img)


if __name__ == "__main__":
    main()
