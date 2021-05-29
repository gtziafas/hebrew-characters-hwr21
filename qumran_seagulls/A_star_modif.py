# from persistence1d import RunPersistence
import cv2
import numpy as np
from PIL import Image, ImageDraw

from qumran_seagulls.persistence1d import RunPersistence

min_persistence = 150
C = 250


def get_sorted_minima(image):
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

    def __eq__(self, other):
        return self.position == other.position


def blocker_dist(child,image):
    d_y = []
    for new_pos in [1,-1]:
        i = 1
        while True:
            if image[child.position[0],child.position[1]+new_pos*i] == 1 or (i + 1 >= len(image) or i + 1 < 0):
                d_y.append(i)
                break
            i += 1

    D = C/(1+np.min(d_y))
    return D


class ContinueIt(Exception):
    pass


def astar(image, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.d = start_node.g = start_node.h = start_node.f = start_node.n = 0
    end_node = Node(None, end)
    end_node.d = end_node.g = end_node.h = end_node.f = end_node.n = 0
    print(end_node.position)
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

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)
        print("closed: ",current_node.position)

        # Found the goal
        if current_node == end_node:
            print("same")
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        child_num=0
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

          # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(image) - 1) or node_position[0] < 0 or node_position[1] > (len(image[len(image)-1]) -1) or node_position[1] < 0:
                print("beyond range")
                continue

            # Make sure walkable terrain or in closed list
            if image[node_position[0]][node_position[1]] != 0:
                print("not walkable")
                continue


            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)
            child_num += 1


        if child_num == 0:
            new_node = Node(current_node, (current_node.position[0]+1, current_node.position[1]))
            children.append(new_node)
            print("must cut through line")

        # Loop through children
        for child in children:
            moveOn=0
            # Child is on the closed list
            for closed_child in closed_list:   # can be removed since checked in line 99
                if child == closed_child:
                    moveOn = 1
                    break
            if moveOn == 1:
                continue
            # Create the f, g, and h values
            if (child.position[1] - current_node.position[1] != 0 and child.position[0] - current_node.position[0] != 0):
                child.n = 14
            else:
                child.n = 10
            child.h = ((child.position[0] - end_node.position[0])**2) + ((child.position[1] - end_node.position[1])**2)
            child.d = blocker_dist(child,image)
            # child.n = current_node.n + cost
            #child.g = child.n + child.d
            # child.f = child.g + child.h
            child.g = current_node.g + 1
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node:
                    moveOn = 1
                    if child.g < open_node.g:
                        open_node.position=child.position
                        open_node.parent = child.parent
                        open_node.g = child.g
                        open_node.f = child.f
                    break

            if moveOn==1:
                continue

            # Add the child to the open list
            open_list.append(child)


def draw_line(example_img_path, path):
    im = Image.open(example_img_path)
    d = ImageDraw.Draw(im)

    for p in path:
        d.line(p, width=1)

    im.save("P123-Fg001-R-C01-R01.jpeg")


def main():
    example_img_path = r"../data/images/P123-Fg001-R-C01-R01-binarized.jpg"
    image = (255 - cv2.imread(str(example_img_path), cv2.IMREAD_GRAYSCALE))/255
    minima = get_sorted_minima(image)
    image = (255 - cv2.imread(str(example_img_path), cv2.IMREAD_GRAYSCALE))
    all_path=[]
    for pos in minima[3:4]:
        start = (0, pos)
        end = (image.shape[0]-1,pos)
        path = astar(image, start, end)
        print(path)
        all_path.append(path)
    draw_line(example_img_path,all_path)


if __name__ == '__main__':
    main()



