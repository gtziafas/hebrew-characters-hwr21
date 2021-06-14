import os

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from qumran_seagulls.models.cnn import BaselineCNN, default_cnn_monkbrill
from qumran_seagulls.persistence1d import RunPersistence

CNN_PATH = "../../data/saved_models/baseline.pt"
N_CLASSES = 27  # I think? Why is it so hard to get the size of the output without running the CNN
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


def plot_sliding_window(line_img: np.ndarray, cnn: BaselineCNN, step_size: int = 10):
    h, w = np.shape(line_img)
    predictions = get_sliding_window_probs(line_img, cnn, step_size)

    max_probs = np.max(predictions[1:, :], axis=0)
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


def main():

    file_id = "P106-Fg002-R-C01-R01"
    saved_cnn = default_cnn_monkbrill()
    saved_cnn.load_state_dict(torch.load(CNN_PATH))

    for i in range(1, 14):
        plt.figure(i)
        example_img = 0xff - cv2.imread(str(f"../../data/lines_cropped/{file_id}/line_{i}.jpg"), cv2.IMREAD_GRAYSCALE)
        char_imgs = plot_sliding_window(example_img, saved_cnn, step_size=10)
        for idx, char_img in enumerate(char_imgs):
            print(f"line: {i} char: {idx} shape{char_img.shape}")
            plt.imshow(char_img)
            dest_dir = f"../../data/chars_cropped/{file_id}/line_{i:02}/"
            os.makedirs(dest_dir, exist_ok=True)
            plt.imsave(dest_dir + f"char_{idx:03}.jpg", char_img)
            plt.show()
    plt.show()


if __name__ == "__main__":
    main()

# to run use command python3 -m qumran_seagulls.scripts.plot_sliding_window_probs