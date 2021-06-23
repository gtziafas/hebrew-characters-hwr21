import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import os

from qumran_seagulls.types import *
from qumran_seagulls.preprocess.shared_astar_funcs.persistence1d import RunPersistence


def get_sorted_minima(image: np.array, min_persistence, axis) -> List[int]:
    histogram = np.sum(image, axis=axis)
    extrema = RunPersistence(histogram)
    minima = extrema[0::2]  # odd elements are minima
    filtered_minima = [t[0] for t in minima if t[1] > min_persistence]
    sorted_minima = sorted(filtered_minima)
    return sorted_minima


def blocker_dist(child, image):
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
