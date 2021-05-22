import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from qumran_seagulls.models.cnn import BaselineCNN, default_cnn_monkbrill

CNN_PATH = "../../data/saved_models/baseline.pt"
N_CLASSES = 27  # I think? Why is it so hard to get the size of the output without running the CNN
input_dim = (75, 75)  # same


def get_roi(window) -> np.ndarray:
    # Here we would do the center of gravity, padding, other fun stuff
    # Return a square image please
    return window


def plot_sliding_window(img: np.ndarray, cnn: BaselineCNN, step_size: int = 10):
    h, w = np.shape(img)
    predictions = np.zeros((N_CLASSES + 1, int(w / step_size) + 1))  # N_CLASSES + 1 because we will use the first row for the window position

    np.set_printoptions(edgeitems=30, linewidth=100000,
                        formatter=dict(float=lambda x: "%.3d" % x))
    window_ctr = 0

    for window_right_edge in range(w, h, -step_size):  # starting from the right, we go left, until we hit h, i.e. the last square window
        window = img[0:h, window_right_edge - h]  # crop the window
        roi = get_roi(window)
        resized_roi = cv2.resize(roi, input_dim, interpolation=cv2.INTER_AREA)

        predictions[0, int(window_right_edge / step_size)] = window_right_edge  # this is needed for the plot

        # print(torch.tensor(resized_roi).unsqueeze(0).shape)
        # predictions[1:, window_ctr] = cnn(torch.tensor(resized_roi).unsqueeze(0))

        window_ctr += 1  # could be more complicated, I just used a second counter

    print(f"predictions matrix:\n{predictions}")

    plt.imshow(img)
    for cls in range(N_CLASSES):
        plt.plot(predictions[0], h - predictions[cls+1])
    plt.show()


def main():
    example_img_path = "../../data/lines_cropped/P106-Fg002-R-C01-R01/line_5.jpg"
    example_img = (255 - cv2.imread(str(example_img_path), cv2.IMREAD_GRAYSCALE)) / 255

    saved_cnn = default_cnn_monkbrill()
    saved_cnn.load_state_dict(torch.load(CNN_PATH))

    plot_sliding_window(example_img, saved_cnn, step_size=10)


if __name__ == "__main__":
    main()
