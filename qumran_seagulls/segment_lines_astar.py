import numpy
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt

debug = True

def segment_img(image):
    histogram = numpy.sum(image, axis=1)
    if debug:
        plt.imshow(image)
        plt.plot(histogram, range(len(histogram)))


    if debug:
        plt.show()


def main():
    example_img_path = "../data/images/P123-Fg001-R-C01-R01-binarized.jpg"
    example_img = (255 - cv2.imread(str(example_img_path), cv2.IMREAD_GRAYSCALE)) / 255
    segment_img(example_img)


if __name__ == "__main__":
    main()
