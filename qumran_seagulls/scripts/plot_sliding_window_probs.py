import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from qumran_seagulls.models.cnn import BaselineCNN, default_cnn_monkbrill
from ..persistence1d import RunPersistence

CNN_PATH = "data/saved_models/baseline.pt"
N_CLASSES = 27  # I think? Why is it so hard to get the size of the output without running the CNN
input_dim = (75, 75)  # same
show_max = True  # show only the max prob in each point
min_persistence = 0.4


def get_roi(window) -> np.ndarray:
    # Here we would do the center of gravity, padding, other fun stuff
    # Return a square image please
    return window


def plot_sliding_window(img: np.ndarray, cnn: BaselineCNN, step_size: int = 10):
    h, w = np.shape(img)
    predictions = np.zeros((N_CLASSES + 1, int(w / step_size) + 1))  # N_CLASSES + 1 because we will use the first row for the window position

    np.set_printoptions(edgeitems=30, linewidth=100000,
                        formatter=dict(float=lambda x: "%+.3f" % x))

    for window_right_edge in range(w, h, -step_size):  # starting from the right, we go left, until we hit h, i.e. the last square window
        predictions[0, int(window_right_edge / step_size)] = window_right_edge  # window position, this is the x position needed for the plot

        window = img[0:h, window_right_edge - h:window_right_edge]  # crop the window
        y = cnn.predict_scores(imgs=[window], device='cpu').softmax(dim=-1)
        print(y)
        predictions[1:, int(window_right_edge / step_size)] = y.detach()

    # delete columns that contain all 0s (first few cols on the left)
    idx = np.argwhere(np.all(predictions[..., :] == 0, axis=0))
    predictions = np.delete(predictions, idx, axis=1)

    print(f"predictions matrix:\n{predictions}")

    plt.imshow(img)
    if show_max:
        max_probs = np.max(predictions[1:, :], axis=0)
        plt.plot(predictions[0] - h / 2, h * 2 - h * max_probs)

        print("max_probs")
        print(max_probs)

        extrema = RunPersistence(max_probs)
        minima = extrema[0::2]  # odd elements are minima
        print(minima)
        filtered_minima = [t[0] for t in minima if t[1] > min_persistence]
        sorted_minima = sorted(filtered_minima)
        print(sorted_minima)

        plt.plot([s * step_size + h/2 for s in sorted_minima] , h*2 - h * max_probs[sorted_minima], "x")
        plt.vlines(x=[s * step_size + h/2 for s in sorted_minima], ymin=0, ymax=2*h, color="green")
    else:
        for cls in range(N_CLASSES):
            plt.plot(predictions[0] - h / 2, h * 2 - h * predictions[cls + 1])
            # x pos: right edge of window - half the height (since window is square) will give the center of the window
            # y pos: imshow flips axes so there is a minus in front of the predictions, scale it up by h and move it below the image
    plt.ylim(ymin=0, ymax=2 * h)
    plt.yticks([0, h, 2 * h])
    plt.gca().invert_yaxis()
    plt.show()


def main():
    example_img_path = "data/lines_cropped/P106-Fg002-R-C01-R01/line_5.jpg"
    example_img = 0xff - cv2.imread(str(example_img_path), cv2.IMREAD_GRAYSCALE)

    saved_cnn = default_cnn_monkbrill()
    saved_cnn.load_state_dict(torch.load(CNN_PATH))

    print(example_img)
    plot_sliding_window(example_img, saved_cnn, step_size=10)


if __name__ == "__main__":
    main()
