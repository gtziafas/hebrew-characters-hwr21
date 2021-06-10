import os
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import glob
import itertools
import numpy as np
from PIL import Image


class Augmenter:
    def __init__(self):
        self.samples = []
        self.path_habbakuk_font = 'dump/'
        self.path_monkbril = '/home/niels/Documents/UNI/Master/Hand Writing Recognition/hebrew-characters-hwr21/data/monkbrill'
        self.images = self.load_habbakuk()
        self.clear = lambda: os.system('clear')
        # we want 300 samples per class
        self.samples_per_class = 300
        # imgaug augmentations:
        self.elastic_transform = iaa.ElasticTransformation(alpha=(5, 30), sigma=(3, 7))
        self.perspective_transform = iaa.PerspectiveTransform(scale=(0.05, 0.10))
        self.affine_transform = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, scale=(0.5, 1.5))
        # cv2 augmentations:
        self.kernel_range_dilate_dilate = (5, 10)
        self.kernel_range_dilate_erode = (4, 6)

    '''
    Load a list containing tuples: (name, image, amount of samples in Monkbrill)
    '''

    def load_habbakuk(self):
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

    def load_monkbrill(self):

        for subdir, dirs, files in os.walk(self.path_monkbril):
            for file in files:
                im = Image.open(os.path.join(subdir, file))
                im = np.array(im, dtype=np.uint8)
                im = cv2.resize(im, (70, 70))
                self.samples.append(im)

    '''
    Erode 1 image with a random kernel drawn from the kernel range
    '''

    def erode(self, image):
        low, high = self.kernel_range_dilate_erode
        # draw random kernel from kernel range
        kernel_size = np.random.randint(low, high)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        augmented = cv2.erode(image, kernel, iterations=2)
        return augmented

    '''
    Dilate 1 image with a random kernel drawn from the kernel range
    '''

    def dilate(self, image):
        low, high = self.kernel_range_dilate_dilate
        # draw random kernel from kernel range
        kernel_size = np.random.randint(low, high)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        augmented = cv2.dilate(image, kernel, iterations=1)
        return augmented

    ''''
    Perform elastic transform on 1 image with a random alpha and sigma drawn from the alpha and sigma ranges
    '''

    def elastic(self, image):
        augmented = self.elastic_transform(image=image)
        return augmented

    ''''
    Perform perspective transform on 1 image with a random scale drawn from the scale range
    '''

    def perspective(self, image):
        augmented = self.perspective_transform(image=image)
        return augmented

    ''''
    Perform affine transform on 1 image with a random scale drawn from the scale range, and translate 
    x- and y-axis independently according to the x/y translation scale
    '''

    def affine(self, image):
        augmented = self.affine_transform(image=image)
        return augmented

    '''
    Perform elastic transform followed by perspective transform
    '''

    def elastic_perspective(self, image):
        augmented = self.elastic(image)
        augmented = self.perspective(augmented)
        return augmented

    '''
    Perform dilation followed by elastic transform
    '''

    def dilate_elastic(self, image):
        augmented = self.dilate(image)
        augmented = self.elastic(augmented)
        return augmented

    '''
    Perform erosion followed by elastic transform
    '''

    def erode_elastic(self, image):
        augmented = self.erode(image)
        augmented = self.elastic(augmented)
        return augmented

    '''
    Perform dilation followed by affine transform
    '''

    def dilate_affine(self, image):
        augmented = self.dilate(image)
        augmented = self.affine(augmented)
        return augmented

    '''
    Perform erosion followed by affine transform
    '''

    def erode_affine(self, image):
        augmented = self.erode(image)
        augmented = self.affine(augmented)
        return augmented

    '''
    Create a total of 300 different samples (including monkbrill samples)
    self.samples will contain all monkbrill samples and all augmenten samples (300 per class)
    
    Atm I also write all generated samples to a augmented/ The amount of samples will differ
    depending on the amount of samples that are already available from monkbrill 
    '''

    def augment(self):

        augmentations = [self.erode, self.dilate, self.affine, self.elastic, self.perspective, self.erode_affine,
                         self.dilate_affine, self.erode_elastic, self.dilate_elastic]
        self.load_monkbrill()

        if not os.path.exists(f'augmented/'):
            os.mkdir(f'augmented/')

        for item in self.images:
            image, name, samples = item
            cnt = 1
            cycle = 1
            for augmentation in itertools.cycle(augmentations):
                augmented = augmentation(image=image)
                augmented = cv2.resize(augmented, (70, 70))
                augmented = cv2.bitwise_not(augmented)
                aug_name = str(augmentation.__name__).replace('self.', '')

                if not os.path.exists(f'augmented/{name}'):
                    os.mkdir(f'augmented/{name}')
                cv2.imwrite(f'augmented/{name}/{name}_{aug_name}_{cycle}.png', augmented)
                self.samples.append(augmented)

                samples += 1
                print(f'samples: {samples}, name: {name}')
                self.clear()

                if cnt == len(augmentations):
                    cnt = 1
                    cycle += 1
                if samples >= self.samples_per_class:
                    break

                cnt += 1


        print("All samples generated successfully")
        return self.samples


def main():
    A = Augmenter()
    A.augment()


if __name__ == '__main__':
    main()
