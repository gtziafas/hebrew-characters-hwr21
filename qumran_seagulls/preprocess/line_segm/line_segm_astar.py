import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # progress bar

from qumran_seagulls.preprocess.shared_astar_funcs.astar_funcs import *
from qumran_seagulls.utils import thresh_invert

min_persistence = 110
debug = False


def call_lineSeg(image):
    cropped_lines = main(image)
    return cropped_lines


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
                        # print("beyond range")
                        pass
                    continue

                #if the path search moves too low, abondon it
                if node_position[1] > start_node.position[1] + avg_dist/2:
                    continue


                # Make sure walkable terrain or in closed list, may not cut through unless the search path has moved into the next line
                if image[node_position[1],node_position[0]] != 0:
                    if node_position[1] > start_node.position[1] - avg_dist/2:
                        if debug:
                            # print("not walkable")
                            pass
                        continue

                # Create new node
                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)
                child_num += 1

            # adding all nodes and assigning extra cost for an ink cut later
            if child_num == 0:
                new_node = Node(current_node, (current_node.position[0] + 1, current_node.position[1]))
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
                child.v = np.abs(child.position[1] - start_node.position[1])  # cost for vertical movement

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


def segment_img(image):
    h, w = np.shape(image)

    # P423-1-Fg002-R-C02-R01-binarized.jpg touches the edge, we don't want that as it breaks A*
    # so just add some 15px padding
    if np.sum(image[:, 0]) > 10:
        image = np.concatenate((np.zeros((h, 15)), image), axis=1)
        h, w = np.shape(image)
        if debug:
            print("added padding")
            plt.imshow(image)
            plt.show()
    minima = get_sorted_minima_scaled(image, min_persistence=min_persistence, axis=1)
    all_paths = []

    print(f"Identified {len(minima)} lines. Image width: {w}. Computing segmentation paths...")

    # adding extra line in path right at the end of the whitespace at the top of the img
    ctr = 0
    for i in range(h):
        if np.sum(image[i, :]) <= 1.0:
            ctr += 1
        else:
            break

    path = []
    for i in range(w):
        path.append(tuple([i, ctr]))
    all_paths.append(path)

    for pos in range(1, len(minima)):
        print(f"Computing path for line {pos}/{len(minima)}...")
        start = (0, minima[pos])
        end = (w - 1, minima[pos])
        avg_dist = (minima[pos] - minima[pos - 1]) / 2
        path = astar(image, start, end, avg_dist)
        if debug:
            print(path)
            print(start, end)
        all_paths.append(path)
    return all_paths


def main(image):
    example_img = thresh_invert(image)/255
    paths = segment_img(example_img)
    if debug:
        plot_lines(example_img, paths)
    cropped_lines = crop_lines(example_img, paths, debug=debug)
    cropped_lines_tight = [crop_out_whitespace(img) for img in cropped_lines if img is not None]

    return [thresh_invert(line.astype(np.uint8)) for line in cropped_lines_tight]


def oldmain(example_img_path):
    example_img = (255 - cv2.imread(str(example_img_path), cv2.IMREAD_GRAYSCALE))/255
    paths = segment_img(example_img)
    if debug:
        # draw_lines(example_img_path, paths, dirname="extracted_images")
        plot_lines(example_img, paths)
    cropped_lines = crop_lines(example_img, paths, debug=debug)
    cropped_lines_tight = [crop_out_whitespace(img) for img in cropped_lines if crop_out_whitespace(img) is not None]
    cropped_lines_dir_path = os.path.splitext('data/extracted_images/' + os.path.split(example_img_path)[1])[0].replace('-binarized','')

    if not os.path.exists(cropped_lines_dir_path):
        os.makedirs(cropped_lines_dir_path, exist_ok=True)
    for idx, cropped_line in enumerate(cropped_lines_tight):
        filename = cropped_lines_dir_path + "/line_" + str(idx) + ".jpg"
        cv2.imwrite(filename, 255 - cropped_line)
    return cropped_lines_tight


if __name__ == '__main__':
    oldmain(sys.argv[1])


# command:
# python3 -m qumran_seagulls.preprocess.line_segm.line_segm_astar data/images/P106-Fg002-R-C01-R01-binarized.jpg
# from project root folder
