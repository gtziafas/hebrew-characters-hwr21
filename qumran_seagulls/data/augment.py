import os
import sys

import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import glob
import itertools
import numpy as np
from PIL import Image


class Augmenter:
    def __init__(self, dataset='habbakuk'):
        self.class_labels_chars = {
            'Alef': 0,
            'Ayin': 1,
            'Bet': 2,
            'Dalet': 3,
            'Gimel': 4,
            'He': 5,
            'Het': 6,
            'Kaf': 7,
            'Kaf-final': 8,
            'Lamed': 9,
            'Mem': 10,
            'Mem-medial': 11,
            'Nun-final': 12,
            'Nun-medial': 13,
            'Pe': 14,
            'Pe-final': 15,
            'Qof': 16,
            'Resh': 17,
            'Samekh': 18,
            'Shin': 19,
            'Taw': 20,
            'Tet': 21,
            'Tsadi-final': 22,
            'Tsadi-medial': 23,
            'Waw': 24,
            'Yod': 25,
            'Zayin': 26

        }
        self.class_labels_styles = {
            'Archaic': 0,
            'Hasmonean': 1,
            'Herodian': 2
        }
        self.dataset = dataset
        self.samples = []
        self.path_habbakuk_font = 'dump/'
        self.path_monkbril = '/home/niels/Documents/UNI/Master/Hand Writing Recognition/hebrew-characters-hwr21/data/monkbrill'
        self.path_styles = '/home/niels/Documents/UNI/Master/Hand Writing Recognition/hebrew-characters-hwr21/data/styles/characters'
        self.images = self.load_styles() if dataset is 'styles' else self.load_habbakuk()
        self.clear = lambda: os.system('clear')
        # we want 300 samples per class
        self.samples_per_class = 50
        # imgaug augmentations:
        self.elastic_transform = iaa.ElasticTransformation(alpha=(5, 30), sigma=(3, 7))
        self.perspective_transform = iaa.PerspectiveTransform(scale=(0.05, 0.10))
        self.affine_transform = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, scale=(0.5, 1.5))
        # cv2 augmentations:
        self.kernel_range_dilate_dilate = (5, 10)
        self.kernel_range_dilate_erode = (4, 6)

    '''
    Load the samples from styles folder, as a list containing tuples: (name, image, amount of samples in Monkbrill)'''

    def load_styles(self):
        ia.seed(1)
        image_list = {'Archaic': [], 'Hasmonean': [], 'Herodian': []}
        image_list = self.load_from_subdir('Archaic', image_list)
        image_list = self.load_from_subdir('Hasmonean', image_list)
        image_list = self.load_from_subdir('Herodian', image_list)
        return image_list

    '''
    Load a list containing tuples: (name, image, amount of samples in Monkbrill)
    '''

    def load_habbakuk(self):
        ia.seed(1)
        image_list = []
        for filename in glob.glob(self.path_habbakuk_font + '/*.png'):
            image_list.append(self.load(filename)
                              )
        return image_list

    '''
    Load all samples from the monkbrill dataset
    '''

    def load_monkbrill(self):
        for subdir, dirs, files in os.walk(self.path_monkbril):
            for file in files:
                im = Image.open(os.path.join(subdir, file))
                im = np.array(im, dtype=np.uint8)
                im = cv2.resize(im, (70, 70))

                label = self.class_labels_chars[subdir.split('monkbrill/', 1)[1]]
                self.samples.append((im, label))

    '''
    A helper function that is used to load and convert images
    '''

    def load(self, filename, style=None, fn=None):
        im = Image.open(filename)
        im = np.array(im, dtype=np.uint8)
        im = cv2.resize(im, (200, 200))  # larger images seem to result in better augmentation
        im = cv2.bitwise_not(im)  # invert image to solve imgaug zero padding problem
        if self.dataset is 'habbakuk':
            filename = filename.replace('.png', '').replace('dump/', '')
            samples = len([item for item in os.listdir(f'{self.path_monkbril}/{filename}')])
            return im, filename, samples
        elif self.dataset is 'styles':
            filename = fn.split('_')[0]
            samples = len([item for item in os.listdir(f'{self.path_styles}/{style}/{filename}')])
            # add samples to self.samples
            self.samples.append((cv2.resize(im, (70, 70)), self.class_labels_styles[style]))
            return im, filename, samples
        else:
            sys.exit('No dataset specified')

    def load_from_subdir(self, style, image_list):
        for subdir, dirs, files in os.walk(f'{self.path_styles}/{style}'):
            for dir in dirs:
                for subdir2, dirs2, files2 in os.walk(f'{self.path_styles}/{style}/{dir}'):
                    for file in files2:
                        image_list[style].append(self.load(f'{self.path_styles}/{style}/{dir}/{file}', style, file))
        return image_list

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

    ''''
        Perform elastic transform on 1 image with a random alpha and sigma drawn from the alpha and sigma ranges
        '''

    def elastic_affine(self, image):
        augmented = self.elastic_transform(image=image)
        augmented = self.affine(augmented)
        return augmented

    ''''
    Perform perspective transform on 1 image with a random scale drawn from the scale range
    '''

    def perspective_affine(self, image):
        augmented = self.perspective_transform(image=image)
        augmented = self.affine(augmented)
        return augmented

    '''
    Perform elastic transform followed by perspective transform
    '''

    def elastic_perspective_affine(self, image):
        augmented = self.elastic(image)
        augmented = self.perspective(augmented)
        augmented = self.affine(augmented)
        return augmented

    '''
    Perform dilation followed by elastic transform
    '''

    def dilate_elastic_affine(self, image):
        augmented = self.dilate(image)
        augmented = self.elastic(augmented)
        augmented = self.affine(augmented)
        return augmented

    '''
    Perform erosion followed by elastic transform
    '''

    def erode_elastic_affine(self, image):
        augmented = self.erode(image)
        augmented = self.elastic(augmented)
        augmented = self.affine(augmented)
        return augmented

    def check_break(self, augmentations, cnt, cycle, samples):
        samples += 1
        if cnt == len(augmentations):
            cnt = 1
            cycle += 1
        if samples >= self.samples_per_class:
            return True, augmentations, cnt, cycle, samples
        cnt += 1

        return False, augmentations, cnt, cycle, samples

    def finalize_augmented_image(self, augmentation, image):
        augmented = augmentation(image=image)
        augmented = cv2.resize(augmented, (70, 70))
        augmented = cv2.bitwise_not(augmented)
        aug_name = str(augmentation.__name__).replace('self.', '')
        return augmented, aug_name
    '''
    Create a total of 300 different samples (including monkbrill samples)
    self.samples will contain all monkbrill samples and all augmenten samples (300 per class)
    
    Atm I also write all generated samples to a augmented/ The amount of samples will differ
    depending on the amount of samples that are already available from monkbrill 
    '''

    def augment(self):

        augmentations = [self.erode, self.dilate, self.affine, self.elastic, self.perspective, self.erode_affine,
                         self.dilate_affine, self.erode_elastic, self.dilate_elastic, self.elastic_affine,
                         self.perspective_affine, self.elastic_perspective_affine, self.dilate_elastic_affine,
                         self.erode_elastic_affine]

        '''
        TODO: Fix load monbrill, Ik wil dat de ist met samples een list van tuples wordt: (image, label)
        Daarnaast wil ik ook tellen hoeveel samples er in een class zitten van styles,  net als ik dat doe bij monkbrill
        en dan aan de hand daarvan een bepaald aantal samples genereren
        '''

        if self.dataset is 'habbakuk':
            self.load_monkbrill()  # for habbakuk we need to load the samples separately like this

            if not os.path.exists(f'augmented_habbakuk/'):
                os.mkdir(f'augmented_habbakuk/')

            for item in self.images:
                image, name, samples = item
                cnt = 1
                cycle = 1
                for augmentation in itertools.cycle(augmentations):
                    # augment, resize, invert, create name
                    augmented, aug_name = self.finalize_augmented_image(augmentation, image)

                    # create dir if not exists
                    if not os.path.exists(f'augmented_habbakuk/{name}'):
                        os.mkdir(f'augmented_habbakuk/{name}')
                    cv2.imwrite(f'augmented_habbakuk/{name}/{name}_{aug_name}_{cycle}.png', augmented)

                    # add sample to the dataset
                    self.samples.append((augmented, self.class_labels_chars[name]))

                    print(f'samples: {samples}, name: {name}')
                    self.clear()
                    # update variables and check termination criteria
                    stop, augmentations, cnt, cycle, samples = self.check_break(augmentations, cnt, cycle, samples)
                    if stop:
                        break

        if self.dataset is 'styles':

            if not os.path.exists(f'augmented_styles/'):
                os.mkdir(f'augmented_styles/')
            for style in self.images:

                for item in self.images[style]:
                    image, name, samples = item
                    cnt = 1
                    cycle = 1
                    print(name)
                    for augmentation in itertools.cycle(augmentations):
                        # augment, resize, invert, create name
                        augmented, aug_name = self.finalize_augmented_image(augmentation, image)

                        # create dir if not exists
                        if not os.path.exists(f'augmented_styles/{style}/{name}'):
                            os.makedirs(f'augmented_styles/{style}/{name}')
                        cv2.imwrite(f'augmented_styles/{style}/{name}/{name}_{aug_name}_{cycle}.png', augmented)

                        # add sample to the dataset
                        self.samples.append((augmented, self.class_labels_styles[style]))

                        print(f'style: {style}, samples: {samples}, name: {name}')
                        self.clear()
                        # update variables and check termination criteria
                        stop, augmentations, cnt, cycle, samples = self.check_break(augmentations, cnt, cycle, samples)
                        if stop:
                            break

        print("All samples generated successfully")
        return self.samples


def main():
    A = Augmenter()
    A.augment()


if __name__ == '__main__':
    main()
