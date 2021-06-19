import os

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import *
from qumran_seagulls.models.cnn import BaselineCNN, monkbrill_with_between_class
from qumran_seagulls.persistence1d import RunPersistence
from scipy.ndimage import gaussian_filter1d

CNN_PATH = "../../data/saved_models/segmenter.pt"
N_CLASSES = 28  # I think? Why is it so hard to get the size of the output without running the CNN
input_dim = (75, 75)  # same
show_max = True  # show only the max prob in each point
min_persistence = 0.4


def crop_characters_from_line(line_img: np.ndarray, coords: List[int]) -> List[np.ndarray]:
    """
    Given an image and a set of x coordinates, crops rectangles between the coordinates
    :param line_img: image to be split
    :param coords: coordinates between characters; MUST BE SORTED
    :return: list of images
    """
    character_images = []
    for i in range(len(coords) - 1):
        character_images.append(line_img[:, coords[i]:coords[i+1]])
    return character_images


def get_sliding_window_probs(img: np.ndarray, cnn: BaselineCNN, step_size: int = 10) -> np.ndarray:
    h, w = np.shape(img)
    predictions = np.zeros((N_CLASSES + 1, int(w / step_size) + 1))  # N_CLASSES + 1 because we will use the first row for the window position

    np.set_printoptions(edgeitems=30, linewidth=100000,
                        formatter=dict(float=lambda x: "%+.3f" % x))

    for window_right_edge in range(w, h, -step_size):  # starting from the right, we go left, until we hit h, i.e. the last square window
        predictions[0, int(window_right_edge / step_size)] = window_right_edge  # window position, this is the x position needed for the plot

        window = img[0:h, window_right_edge - h:window_right_edge]  # crop the window
        y = cnn.predict_scores(imgs=[window], device='cpu').softmax(dim=-1)
        # print(y)
        predictions[1:, int(window_right_edge / step_size)] = y.detach()

    # delete columns that contain all 0s (first few cols on the left)
    idx = np.argwhere(np.all(predictions[..., :] == 0, axis=0))
    predictions = np.delete(predictions, idx, axis=1)

    # print(f"predictions matrix:\n{predictions}")

    return predictions


def plot_sliding_window(line_img: np.ndarray, cnn: BaselineCNN, step_size: int = 10, to_remove: Tuple[int, int] = (0, 0)):
    h, w = np.shape(line_img)
    predictions = get_sliding_window_probs(line_img, cnn, step_size)

    # discard 1st row (index) and last row (probability of class "between")
    max_probs = np.max(predictions[1:-1, :], axis=0)
    # print("line")

    # print("max_probs")
    # print(max_probs)

    extrema = RunPersistence(max_probs)
    minima = extrema[0::2]  # odd elements are minima
    print(minima)
    filtered_minima = [t[0] for t in minima if t[1] > min_persistence]
    sorted_minima = sorted(filtered_minima)
    print(sorted_minima)

    sorted_minima_pixel_coords = [int(s * step_size + h/2) for s in sorted_minima]

    character_images = crop_characters_from_line(line_img, sorted_minima_pixel_coords)

    plt.imshow(line_img)
    if show_max:
        plt.plot(predictions[0] - h / 2, h * 2 - h * max_probs)

        plt.ylim(ymin=0, ymax=2 * h)
        plt.yticks([0, h, 2 * h])
        plt.gca().invert_yaxis()

        plt.plot(sorted_minima_pixel_coords, h * 2 - h * max_probs[sorted_minima], "x")
        plt.vlines(x=sorted_minima_pixel_coords, ymin=0, ymax=2 * h, color="green")
    else:
        for cls in range(N_CLASSES):
            plt.plot(predictions[0] - h / 2, h * 2 - h * predictions[cls + 1])
            plt.ylim(ymin=0, ymax=2 * h)
            plt.yticks([0, h, 2 * h])
            plt.gca().invert_yaxis()
            # x pos: right edge of window - half the height (since window is square) will give the center of the window
            # y pos: imshow flips axes so there is a minus in front of the predictions, scale it up by h and move it below the image
    plt.show()

    return character_images


def get_asc_desc_offsets(line_imgs: List[np.array]):
    """This will identify the top and bottom offsets where ascenders and descenders are, for each line"""

    # project horizontally
    projections = [np.sum(line_image, axis=1) for line_image in line_imgs]
    # smooth the projection
    projections_filtered = [gaussian_filter1d(proj, sigma=6) for proj in projections]

    # identify the mode of the projection
    projection_maxes = [proj.argmax() for proj in projections_filtered]

    # convert to list of tuples so idx can be changed
    projection_points = [list(enumerate(proj)) for proj in projections]

    # align all projections so mode is on 0
    projection_points_centered = [[((x - offset), y) for (x, y) in proj] for (proj, offset) in zip(projection_points, projection_maxes)]

    # convert to dict since the x values are not changing and it's easier to search
    proj_pts_centered_dict = [dict(proj) for proj in projection_points_centered]

    # get all indices so we know how to sum the projections
    x_values = np.unique([x for proj in projection_points_centered for (x, y) in proj])
    y_values = np.zeros_like(x_values)

    # for each x, sum the projection on every image that is on that x
    for idx, x_value in enumerate(x_values):
        for proj in proj_pts_centered_dict:
            y_values[idx] += proj.get(x_value,0)

    # # normalize to [0, 1]
    y_values = y_values.astype('float64')
    y_values /= np.max(y_values)

    print(f"xvals {x_values}\n")
    print(f"yvals {y_values}\n")

    plt.plot(x_values, y_values)
    plt.show()

def main():

    file_id = "P106-Fg002-R-C01-R01"
    saved_cnn = monkbrill_with_between_class()
    saved_cnn.load_state_dict(torch.load(CNN_PATH))

    line_imgs = []
    for i in range(1, 14):
        line_imgs.append(0xff - cv2.imread(str(f"../../data/lines_cropped/{file_id}/line_{i}.jpg"), cv2.IMREAD_GRAYSCALE))

    to_remove = get_asc_desc_offsets(line_imgs)

    # for line_img in line_imgs:
    #     plt.figure(i)
    #     char_imgs = plot_sliding_window(line_img, saved_cnn, step_size=15)
    #     for idx, char_img in enumerate(char_imgs):
    #         print(f"line: {i} char: {idx} shape: {char_img.shape}")
    #         dest_dir = f"../../data/chars_cropped/{file_id}/line_{i:02}/"
    #         os.makedirs(dest_dir, exist_ok=True)
    #         plt.imsave(dest_dir + f"char_{idx:03}.jpg", char_img)
    #         # plt.imshow(char_img)
    #         # plt.show()
    # plt.show()


if __name__ == "__main__":
    main()

# to run use command python3 -m qumran_seagulls.scripts.plot_sliding_window_probs