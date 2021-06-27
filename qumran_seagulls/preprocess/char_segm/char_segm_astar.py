import sys
from tqdm import tqdm  # progress bar

from qumran_seagulls.preprocess.shared_astar_funcs.astar_funcs import *
from qumran_seagulls.types import *
from qumran_seagulls.preprocess.shared_astar_funcs.astar_funcs import get_sorted_minima_with_probs

min_persistence = 30
debug = False


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

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f <= current_node.f:
                current_node = item
                current_index = index
        if debug:
            print("Chosen:"+str(current_node.position)+"Cost:" + str(current_node.f))
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
            if node_position[0] > (image.shape[1] - 1) or node_position[0] < 0 or node_position[1] < 0 or node_position[
                1] > (image.shape[0] - 1):
                if debug:
                    print("beyond range")
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)
            child_num += 1

        # adding all nodes and assigning extra cost for an ink cut later
        if child_num == 0:
            new_node = Node(current_node, (current_node.position[0], current_node.position[1] + 1))
            children.append(new_node)
            if debug:
                print("must cut through line")

        # Loop through children
        for child in children:
            move_on = 0
            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    if debug:
                        print(str(node_position)+"in closed list")
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
            child.v = np.abs(child.position[0] - start_node.position[0])  # cost for horizontal movement
            child.d, child.d2 = blocker_dist(child, image, debug=debug)
            if image[child.position[1], child.position[0]] > 0.9:
                child.m = 1000
            child.g = current_node.g + child.n + child.v + child.m + child.d + child.d2
            child.f = child.g + child.h  # heuristic still needed to speed up computations
            if debug:
                print(child.position, child.f)

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node:
                    move_on = 1
                    if child.f < open_node.f:
                        # open_node.position = child.position
                        open_node.parent = child.parent
                        open_node.g = child.g
                        open_node.f = child.f
                    break

            if move_on == 1:
                continue

            # Add the child to the open list
            open_list.append(child)


def segment_img(image, window_size=(75, 75)):
    h, w = np.shape(image)
    minima = get_sorted_minima_with_probs(image, min_persistence=min_persistence, axis=0, window_size=window_size, debug=debug)
    all_paths = []
    path = []

    print(f"Identified {len(minima)} characters. Image height: {h}. Computing character segmentation paths...")

    # adding extra line in path
    for i in range(image.shape[1]):
        path.append(tuple([i, 1]))
    all_paths.append(path)

    with tqdm(total=len(minima)) as pbar:
        for pos in range(1, len(minima)):
            if debug:
                print(f"Computing path for character {pos}/{len(minima)}...")
            start = (minima[pos], 0)
            end = (minima[pos], h-1)
            avg_dist = (minima[pos] - minima[pos - 1]) / 2
            path = astar(image, start, end, avg_dist)
            if debug:
                print(path)
                print(start, end)
            all_paths.append(path)
            pbar.update(pos - pbar.n + 1)
    return all_paths


def segment_characters(line: array, window_size=(75, 75)) -> List[array]:
    paths = segment_img(line, window_size=window_size)
    cropped_chars = crop_lines(line, paths, debug=False)
    chars = [c for c in cropped_chars if np.sum(c) > 200]
    # TODO: there is an issue where the first segmentation path is wrong:
    chars = [c for c in chars if c.shape[1] < 3/4 * line.shape[1]]  # hack
    print(f"Kept {len(chars)} characters after discarding ink blobs.")
    return chars


def main(img_path):
    img = (255 - cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE))/255
    plt.figure(0)
    plt.imshow(img)
    chars = segment_characters(img)
    for idx, cropped_line in enumerate(chars):
        plt.figure(idx+1)
        plt.imshow(cropped_line)
    plt.show()


def main_with_plotting(img_path):
    example_img = (255 - cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE))/255
    paths = segment_img(example_img)
    if debug:
        draw_lines(img_path, paths, dirname="extracted_char")
        plot_lines(example_img, paths)
    # cropped_lines = crop_lines(example_img, paths, debug=False)
    # cropped_lines_dir_path = os.path.splitext('data/extracted_char/' + os.path.split(img_path)[1])[0]
    #
    # if not os.path.exists(cropped_lines_dir_path):
    #     os.makedirs(cropped_lines_dir_path, exist_ok=True)
    # for idx, cropped_line in enumerate(cropped_lines):
    #     filename = cropped_lines_dir_path + "/char_" + str(idx) + ".jpg"
    #     cv2.imwrite(filename, 255 - cropped_line)


if __name__ == '__main__':
    main_with_plotting(sys.argv[1])


# command:
# python3 -m qumran_seagulls.preprocess.char_segm.char_segm_astar data/extracted_char/P106-Fg002-R-C01-R01-binarized.jpg
# from project root folder
