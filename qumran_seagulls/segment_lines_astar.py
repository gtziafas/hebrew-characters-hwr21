import numpy
import cv2
import matplotlib.pyplot as plt
from persistence1d import RunPersistence

debug = True
min_persistence = 150


def segment_img(image):
    histogram = numpy.sum(image, axis=1)
    if debug:
        plt.imshow(image)
        plt.plot(histogram, range(len(histogram)))

    extrema = RunPersistence(histogram)
    filtered_extrema = [t[0] for t in extrema if t[1] > min_persistence]
    sorted_extrema = sorted(filtered_extrema)

    plt.plot(histogram[sorted_extrema], sorted_extrema, "x")

    if debug:
        plt.show()


def main():
    example_img_path = "../data/images/P123-Fg001-R-C01-R01-binarized.jpg"
    example_img = (255 - cv2.imread(str(example_img_path), cv2.IMREAD_GRAYSCALE)) / 255
    segment_img(example_img)


if __name__ == "__main__":
    main()
