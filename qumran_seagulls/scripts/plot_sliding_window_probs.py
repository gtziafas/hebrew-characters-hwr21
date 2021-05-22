import cv2
import torch
import numpy as np

CNN_PATH = "..data/saved_models/baseline.pt"
N_CLASSES = 27  # I think? Why is it so hard to get the size of the output without running the CNN
input_dim = (75, 75)  # same


def get_roi(window) -> np.ndarray:
    # Here we would do the center of gravity, padding, other fun stuff
    # Return a square image please
    return window


def plot_sliding_window(img: np.ndarray, cnn: torch.nn.Module, step_size: int = 10):
    h, w = np.shape(img)
    predictions = np.ndarray((N_CLASSES, w / step_size))
    window_ctr = 0

    for window_right_edge in range(w, 0, -step_size):
        window = img[0:h, window_right_edge - h]  # crop the window
        roi = get_roi(window)
        resized_roi = cv2.resize(roi, input_dim, interpolation=cv2.INTER_AREA)
        predictions[:, window_ctr] = cnn(resized_roi)
        window_ctr += 1  # could be more complicated, I just used a second counter

    print(predictions)


def main():
    example_img_path = "../data/lines_cropped/P106-Fg002-R-C01-R01/line_5.jpg"
    example_img = (255 - cv2.imread(str(example_img_path), cv2.IMREAD_GRAYSCALE)) / 255
    saved_cnn = torch.load(CNN_PATH)
    plot_sliding_window(example_img, saved_cnn, step_size=10)


if __name__ == "__main__":
    main()
