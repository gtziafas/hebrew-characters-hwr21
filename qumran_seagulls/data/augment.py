import os
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import glob
import numpy as np
from PIL import Image


class Augmenter:
    def __init__(self):
        self.path_habbakuk_font = 'dump/'
        self.path_monkbril = '/home/niels/Documents/UNI/Master/Hand Writing Recognition/hebrew-characters-hwr21/data/monkbrill'
        self.images = self.load_images()
        # imgaug augmentations:
        self.elastic_transform = iaa.ElasticTransformation(alpha=(5, 30), sigma=(3, 7))
        self.perspective_transform = iaa.PerspectiveTransform(scale=(0.05, 0.10))
        self.affine_transform = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, scale=(0.5, 1.5))
        # cv2 augmentations:
        self.kernel_range_dilate_erode = (0, 0)

    '''
    Load a list containing tuples: (name, image, amount of samples in Monkbrill)
    '''

    def load_images(self):
        ia.seed(1)
        image_list = []
        for filename in glob.glob(self.path_habbakuk_font + '/*.png'):
            im = Image.open(filename)
            im = np.array(im, dtype=np.uint8)
            im = cv2.resize(im, (200, 200))  # larger images seem to result in better augmentation
            im = cv2.bitwise_not(im)  # invert image to solve imgaug zero padding problem
            filename = filename.replace('.png', '').replace('dump/', '')
            samples = len([item for item in os.listdir(self.path_monkbril + '/' + filename)])
            image_list.append((im, filename, samples))
        return image_list

    '''
    Erode 1 image with a random kernel drawn from the kernel range
    '''

    def erode(self, image):
        low, high = self.kernel_range_dilate_erode
        # draw random kernel from kernel range
        kernel_size = np.random.randint(low, high)
        kernel = (kernel_size, kernel_size)
        augmented = cv2.erode(image, kernel, iterations=1)
        # revert inversion
        augmented = cv2.bitwise_not(augmented)
        return augmented

    '''
    Dilate 1 image with a random kernel drawn from the kernel range
    '''

    def dilate(self, image):
        low, high = self.kernel_range_dilate_erode
        # draw random kernel from kernel range
        kernel_size = np.random.randint(low, high)
        kernel = (kernel_size, kernel_size)
        augmented = cv2.dilate(image, kernel, iterations=1)
        # revert inversion
        augmented = cv2.bitwise_not(augmented)
        return augmented

    ''''
    Perform elastic transform on 1 image with a random alpha and sigma drawn from the alpha and sigma ranges
    '''

    def elastic(self, image):
        augmented = self.elastic_transform(images=image)
        # revert inversion
        augmented = cv2.bitwise_not(augmented)
        return augmented

    ''''
    Perform perspective transform on 1 image with a random scale drawn from the scale range
    '''

    def perspective(self, image):
        augmented = self.perspective_transform(images=image)
        # revert inversion
        augmented = cv2.bitwise_not(augmented)
        return augmented

    ''''
    Perform affine transform on 1 image with a random scale drawn from the scale range, and translate 
    x- and y-axis independently according to the x/y translation scale
    '''

    def affine(self, image):
        augmented = self.affine_transform(images=image)
        # revert inversion
        augmented = cv2.bitwise_not(augmented)
        return augmented

    '''
    Heavy augmentation: create a total of 300 different samples (including monkbrill samples)
    '''

    def augment(self):

        pass


def main():
    A = Augmenter()
    print(A.images[0][2])


if __name__ == '__main__':
    main()
