import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import interpolate

from qumran_seagulls.models.cnn import monkbrill_with_between_class
from qumran_seagulls.preprocess.char_segm.char_segm_sliding_window_classifier import get_sliding_window_probs, \
    get_sliding_window_probs_with_cropping
from qumran_seagulls.preprocess.shared_astar_funcs.persistence1d import RunPersistence
from qumran_seagulls.types import *


def get_sorted_minima(image: np.array, min_persistence, axis) -> List[int]:
    histogram = np.sum(image, axis=axis)
    extrema = RunPersistence(histogram)
    minima = extrema[0::2]  # odd elements are minima
    filtered_minima = [t[0] for t in minima if t[1] > min_persistence]
    sorted_minima = sorted(filtered_minima)
    return sorted_minima


def get_sorted_minima_scaled(image: np.array, min_persistence, axis) -> List[int]:
    histogram = np.sum(image, axis=axis)

    max_histogram_value = np.max(histogram)
    scaled_min_persistence = min_persistence * max_histogram_value / 1000

    extrema = RunPersistence(histogram)
    minima = extrema[0::2]  # odd elements are minima
    filtered_minima = [t[0] for t in minima if t[1] > scaled_min_persistence and histogram[t[0]] < 0.5 * max_histogram_value]
    sorted_minima = sorted(filtered_minima)
    return sorted_minima


def blocker_dist(child, image, debug):
    d_x = []
    for new_pos in [1, -1]:
        i = 0
        x_cord = child.position[1]
        while (x_cord <= image.shape[1]-1) and x_cord >= 0:
            if image[child.position[1], x_cord] != 0:
                d_x.append(i)
                break
            i += 1
            x_cord += new_pos
        if (x_cord > image.shape[1]-1) or x_cord < 0:
            d_x.append(10000)  # some maximum value

    if debug:
        print(d_x)
    D = 1 / (1+np.min(d_x))
    D2 = 1 / ((1+np.min(d_x))**2)
    return D, D2


class Node:
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0
        self.d = 0
        self.n = 0
        self.m = 0
        self.v = 0
        self.d2 = 0

    def __eq__(self, other):
        return self.position == other.position


def plot_lines(image, paths):
    plt.imshow(image)

    for path in paths:
        plt.plot(*zip(*path))

    plt.show()


def draw_lines(img_path, line_paths, dirname):
    im = Image.open(img_path)
    d = ImageDraw.Draw(im)

    for p in line_paths:
        d.line(p, width=1)

    if not os.path.exists(f'data/{dirname}/'):
        os.mkdir(f'data/{dirname}/')
    save_filename = f"data/{dirname}/" + os.path.split(img_path)[1]
    im.save(save_filename)


def crop_lines(image: np.ndarray, paths: List[List[Tuple[int]]], debug: bool):
    """
    Crops all the lines from the image, given the paths between them
    Based on: https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
    :param image: image to be cropped
    :param paths: list of paths, each path being a list of 2d coordinates
    :param debug: whether to show the cropped images as plots
    :return:
    """

    normalized_img = image * 255

    cropped_lines = []
    l = len(paths)
    for i in range(l - 1):  # we need the current path and the next one
        upper_path = paths[i]
        lower_path = paths[i+1]
        polygon = np.array(upper_path + lower_path[::-1])  # go right on the upper path and left on the lower path
                                                           # to obtain a polygon that doesn't cross itself

        # (1) Crop the bounding rect
        rect = cv2.boundingRect(polygon)
        x, y, w, h = rect
        cropped = normalized_img[y:y + h, x:x + w].copy()

        # (2) make mask
        polygon = polygon - polygon.min(axis=0)

        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [polygon], -1, (255, 255, 255), -1, cv2.LINE_AA)

        # (3) do bit-op
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)

        np.set_printoptions(edgeitems=30, linewidth=100000,
                            formatter=dict(float=lambda x: "%.3d" % x))
        if debug:
            plt.imshow(dst/255)
            plt.show()
        cropped_lines.append(dst)

    return cropped_lines


def get_sorted_minima_with_probs(image: np.array, min_persistence, axis, window_size=(75,75), debug=False) -> List[int]:
    """
    Gets the sorted minima of the histogram, but the histogram is obtained by mixing with the ink projection
    with the probability that the current window is right on a character.
    This function is only usable for character segmentation.
    :param image:
    :param min_persistence:
    :param axis:
    :param debug:
    :return:
    """

    h, w = image.shape
    histogram = np.sum(image, axis=axis)

    saved_cnn = monkbrill_with_between_class()
    saved_cnn.load_state_dict(torch.load("data/saved_models/segmenter.pt"))
    probs = get_sliding_window_probs_with_cropping(image*255, cnn=saved_cnn, step_size=5, window_size=window_size)

    # Discard the first row (window center position) and the last row (probability of "between" class)
    # to get the max probability that the window is on a character
    max_probs = np.max(probs[1:-1, :], axis=0)

    # Interpolate to get values between the positions of the sliding windows
    x_interp_range = np.arange(0, w, step=1, dtype=int)
    max_probs_interp_function = interpolate.interp1d(probs[0, :], max_probs, fill_value=(max_probs[0], max_probs[-1]), bounds_error=False)
    max_probs_interp = max_probs_interp_function(x_interp_range)

    # Scale up to have the same max value as the histogram
    height_scale_function = interpolate.interp1d([np.amin(max_probs_interp), np.amax(max_probs_interp)], [0, np.amax(histogram)])
    max_probs_interp = height_scale_function(max_probs_interp)

    # Mix the histogram with the probabilities
    mixed_histogram = histogram * 0.75 + max_probs_interp * 0.75

    extrema = RunPersistence(mixed_histogram)
    minima = extrema[0::2]  # odd elements are minima
    filtered_minima = [t[0] for t in minima if t[1] > min_persistence]
    sorted_minima = sorted(filtered_minima)

    if debug:
        plt.imshow(image)
        plt.plot(probs[0, :], h * 2 - 100 * max_probs, label="max_probs")
        plt.plot(probs[0, :], h * 2 - 100 * max_probs, ".", label="max_probs")
        plt.plot(h * 2 - histogram, label="Ink proj")
        plt.plot(h * 2 - mixed_histogram, label="mixed_histogram")
        plt.plot(sorted_minima, h*2 - mixed_histogram[sorted_minima], "x", c="red")
        plt.legend()

    return sorted_minima