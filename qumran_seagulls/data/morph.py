import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
import imgaug as ia
import imageio
import glob
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-c', '--create_data', type=str, help='Create Augmented data: t/f', default='f')
parser.add_argument('-l', '--letter', type=int, help='letter of the habbakuk alphabet', default=1)
parser.add_argument('-a', '--augmentation', type=str,
                    help='Augmentation type: Elastic (e), Affine (a), Perspective (p)', default='e')

args = parser.parse_args()

if args.letter > 27:
    parser.error('-l: Integer should be <= 27')
if args.augmentation not in {'a', 'e', 'p'}:
    parser.error('-a: Augmentation should be: a, e or p')
if args.create_data not in {'t', 'f'}:
    parser.error('-c: Should be: t or f')

'''
Load all image samples from the dump folder. 
'''


def load_images():
    ia.seed(1)
    image_list = []
    for filename in glob.glob('dump/*.png'):
        im = Image.open(filename)
        im = np.array(im, dtype=np.uint8)
        image_list.append(im)

    return image_list


'''
Elastic:
alpha controls the strength of the displacement: higher values mean that pixels are moved further.
sigma controls the smoothness of the displacement: higher values lead to smoother patterns.
A relation of 10:1 seems to be good for alpha and sigma

Perspective:
Apply perspective transformations using a random scale between 0.05 and 0.15 per image, where the
scale is roughly a measure of how far the perspective transformation’s corner points may be 
distanced from the image’s corner points.
'''


def augment_images(image_list):
    aug_list = []
    aug = None
    aug_name = None

    if args.augmentation == 'e':
        alpha = input('Alpha = ')
        sigma = input('Sigma = ')
        aug = iaa.ElasticTransformation(alpha=float(alpha), sigma=float(sigma))
        aug_name = 'Elastic'
    elif args.augmentation == 'a':
        r1 = input('Range start = ')
        r2 = input('Range end = ')
        aug = iaa.Affine(scale=(float(r1), float(r2)))
        aug_name = 'Affine'
    elif args.augmentation == 'p':
        r1 = input('Range start = ')
        r2 = input('Range end = ')
        aug = iaa.PerspectiveTransform(scale=(float(r1), float(r2)))
        aug_name = 'Perspective'

    for image in image_list:
        aug_list.append(aug.augment_images(image))

    return aug_list, aug_name


'''
Save all images in: augmented_font_samples
'''


def save_images(image_list, name):
    for idx, image in enumerate(image_list):
        imageio.imwrite(f'augmented_font_samples/{name}_{str(idx)}.png', image)


def show_comparison():
    global im
    idx = 1
    done = 'n'

    ia.seed(1)
    for filename in glob.glob('dump/*.png'):
        if idx == args.letter:
            im = Image.open(filename)
            im = np.array(im, dtype=np.uint8)
            break
        idx += 1
    image_list = [im]

    if args.augmentation == 'e':
        while done == 'n':
            alpha = input('Alpha = ')
            sigma = input('Sigma = ')
            aug_elastic = iaa.ElasticTransformation(alpha=float(alpha), sigma=float(sigma))
            image_list.append(aug_elastic.augment_images(im))
            plot_comparison(image_list)
            done = input('Quit y/n? ')

    elif args.augmentation == 'a':
        while done == 'n':
            r1 = input('Range start = ')
            r2 = input('Range end = ')
            aug_affine = iaa.Affine(scale=(float(r1), float(r2)))
            image_list.append(aug_affine.augment_images(im))
            plot_comparison(image_list)
            done = input('Quit y/n? ')

    elif args.augmentation == 'p':
        while done == 'n':
            r1 = input('Range start = ')
            r2 = input('Range end = ')
            aug_perspective = iaa.PerspectiveTransform(scale=(float(r1), float(r2)))
            image_list.append(aug_perspective.augment_images(im))
            plot_comparison(image_list)
            done = input('Quit y/n? ')


def plot_comparison(image_list):
    fig = plt.figure(figsize=(10, 10))
    for idx, image in enumerate(image_list):
        fig.add_subplot(1, len(image_list), idx + 1)
        plt.imshow(image)
    plt.show()


def main():
    if args.create_data == 't':

        images_list = load_images()
        aug_list, aug_name = augment_images(images_list)
        save_images(aug_list, aug_name)
        print("Augmented images saves successfully!")
    else:
        show_comparison()


if __name__ == '__main__':
    main()
