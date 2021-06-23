# Add data/extracted_images folder
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm  # progress bar
from qumran_seagulls.types import *
import os

from qumran_seagulls.preprocess.shared_astar_funcs.persistence1d import RunPersistence

min_persistence = 170
debug = False


def get_sorted_minima(image: np.array) -> List[int]:
    histogram = np.sum(image, axis=1)
    extrema = RunPersistence(histogram)
    minima = extrema[0::2]  # odd elements are minima
    filtered_minima = [t[0] for t in minima if t[1] > min_persistence]
    sorted_minima = sorted(filtered_minima)
    return sorted_minima


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


# def blocker_dist(child, image):
#     d_y = []
#     for new_pos in [1, -1]:
#         i = 0
#         y_cord = child.position[1]
#         while (y_cord <= image.shape[0]-1) and y_cord >= 0:
#             if image[y_cord, child.position[0]] != 0:
#                 d_y.append(i)
#                 break
#             i += 1
#             y_cord += new_pos
#         if (y_cord > image.shape[0]-1) or y_cord < 0:
#             d_y.append(2709)  # some maximum value
#
#     print(d_y)
#     D = 1 / (1+np.min(d_y))
#     D2 = 1 / ((1+np.min(d_y))**2)
#     return D, D2


def astar(image, start, end, avg_dist):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    end_node = Node(None, end)
    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    with tqdm(total=end[0]) as pbar:
        # Loop until you find the end
        while len(open_list) > 0:

            # Get the current node
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f <= current_node.f:
                    current_node = item
                    current_index = index

            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the goal
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]  # Return reversed path

            # Generate children
            children = []
            child_num = 0
            for new_position in [(0, -1), (0, 1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares
                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Make sure within range
                if node_position[0] > (image.shape[1] - 1) or node_position[0] < start_node.position[0] or node_position[
                    1] > (image.shape[0] - 1) or node_position[1] < 0:
                    if debug:
                        print("beyond range")
                    continue

                #if the path search moves too low, abondon it
                if node_position[1] > start_node.position[1] + avg_dist/2:
                    continue


                # Make sure walkable terrain or in closed list, may not cut through unless the search path has moved into the next line
                if image[node_position[1],node_position[0]] != 0:
                    if node_position[1] > start_node.position[1] - avg_dist/2:
                        if debug:
                            print("not walkable")
                        continue

                # Create new node
                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)
                child_num += 1

            # adding all nodes and assigning extra cost for an ink cut later
            if child_num == 0:
                new_node = Node(current_node, (current_node.position[0]+1, current_node.position[1]))
                children.append(new_node)
                if debug:
                    print("must cut through line")

            # Loop through children
            for child in children:
                move_on = 0
                # Child is on the closed list
                for closed_child in closed_list:
                    if child == closed_child:
                        move_on = 1
                        break
                if move_on == 1:
                    continue
                # Create the f, g, and h values
                if child.position[1] - current_node.position[1] != 0 and child.position[0] - current_node.position[0] != 0:
                    child.n = 14
                else:
                    child.n = 10
                child.h = (np.abs(child.position[0] - end_node.position[0])**2) + (np.abs(child.position[1] - end_node.position[1])**2)
                child.v = np.abs(child.position[1] - start_node.position[1]) #cost for vertical movement

                if image[child.position[1], child.position[0]] != 0:
                    child.m = 25
                child.g = current_node.g + child.n + child.v + child.m
                child.f = child.g + child.h  # heuristic still needed to speed up computations


                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node:
                        move_on = 1
                        if child.g < open_node.g:
                            open_node.position = child.position
                            open_node.parent = child.parent
                            open_node.g = child.g
                            open_node.f = child.f
                        break

                if move_on == 1:
                    continue

                # Add the child to the open list
                open_list.append(child)
                pbar.update(child.position[0] - pbar.n)


def draw_line(example_img_path, path):
    im = Image.open(example_img_path)
    d = ImageDraw.Draw(im)

    for p in path:
        d.line(p, width=1)

    if not os.path.exists('data/extracted_images/'):
        os.mkdir('data/extracted_images/')
    save_filename = r"data/extracted_images/" + os.path.split(example_img_path)[1]
    im.save(save_filename)


def plot_lines(image, paths):
    plt.imshow(image)

    for path in paths:
        plt.plot(*zip(*path))

    plt.show()


def crop_lines(image: np.ndarray, paths: List[List[Tuple[int]]]):
    """
    Crops all the lines from the image, given the paths between them
    Based on: https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
    :param image: image to be cropped
    :param paths: list of paths, each path being a list of 2d coordinates
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


def segment_img(image):
    h, w = np.shape(image)
    minima = get_sorted_minima(image)
    all_paths = []
    path = []

    print(f"Identified {len(minima)} lines. Image width: {w}. Computing paths...")

    #adding extra line in path
    for i in range(image.shape[1]):
        path.append(tuple([i, 1]))
    all_paths.append(path)
    for pos in range(1,len(minima)):

        print(f"Computing path for line {pos}/{len(minima)}...")
        start = (0, minima[pos])
        end = (w - 1, minima[pos])
        avg_dist = (minima[pos] - minima[pos-1])/2
        path = astar(image, start, end, avg_dist)
        if debug:
            print(path)
        all_paths.append(path)
    return all_paths


def main(argv):
    print(argv)
    example_img_path = argv
    example_img = (255 - cv2.imread(str(example_img_path), cv2.IMREAD_GRAYSCALE))/255
    paths = segment_img(example_img)
    if debug:
        draw_line(example_img_path, paths)
        plot_lines(example_img, paths)
    cropped_lines = crop_lines(example_img, paths)

    cropped_lines_dir_path = os.path.splitext('data/extracted_images/' + os.path.split(example_img_path)[1])[0].replace('-binarized','')

    if not os.path.exists(cropped_lines_dir_path):
        os.makedirs(cropped_lines_dir_path, exist_ok=True)
    for idx, cropped_line in enumerate(cropped_lines):
        filename = cropped_lines_dir_path + "/line_" + str(idx) + ".jpg"
        cv2.imwrite(filename, 255-cropped_line)


if __name__ == '__main__':
    main(sys.argv[1])


# command:
# python3 -m qumran_seagulls.preprocess.line_segm.line_segm_astar data/images/P106-Fg002-R-C01-R01-binarized.jpg
# from project root folder
