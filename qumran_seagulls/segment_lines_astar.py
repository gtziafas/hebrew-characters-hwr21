import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from persistence1d import RunPersistence

debug = True
min_persistence = 150


def find_paths(image, minima):
    h, w = np.shape(image)
    inverted_image = 1 - image
    grid = Grid(matrix=inverted_image)

    paths = []

    for m in minima:
        start = grid.node(0, m)
        end = grid.node(w - 1, m)
        print(f"computing path from {(0, m)} to {(w - 1, m)}")
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        try:
            path, runs = finder.find_path(start, end, grid)
            print('operations:', runs, 'path length:', len(path))
            print(list(path))
            # print(grid.grid_str(path=path, start=start, end=end))
            paths.append(path)
            plt.plot(*zip(*path), color="green")
        except ValueError:
            print(f"couldn't find path from {(0, m)} to {(w - 1, m)}")
            pass  # Couldn't segment this line; it's not the end of the world, but it should be fixed
        # break

    return paths


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
    print(f"sorted minima: {sorted_minima}")

    plt.plot(histogram[sorted_minima], sorted_minima, "x")

    # crop_straight_from_minima(image, sorted_minima)
    print("finding paths")

    paths = find_paths(image, sorted_minima)
    plt.show()
    print("finished finding paths")


def main():
    example_img_path = "../data/images/P123-Fg001-R-C01-R01-binarized.jpg"
    example_img = (255 - cv2.imread(str(example_img_path), cv2.IMREAD_GRAYSCALE)) / 255
    segment_img(example_img)


if __name__ == "__main__":
    main()
