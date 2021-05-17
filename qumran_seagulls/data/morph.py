import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
import imgaug as ia
import imageio
import glob

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
    aug_elastic_list = []
    aug_perspective_list = []
    aug_affine_list = []

    aug_elastic = iaa.ElasticTransformation(alpha=(0, 35.0), sigma=(2, 3))
    aug_perspective = iaa.PerspectiveTransform(scale=(0.05, 0.15))
    aug_affine = iaa.Affine(scale=(0.5, 1.5))

    for image in image_list:
        aug_elastic_list.append(aug_elastic.augment_images(image))
        aug_perspective_list.append(aug_perspective.augment_images(image))
        aug_affine_list.append(aug_affine.augment_images(image))

    return aug_elastic_list, aug_perspective_list, aug_affine_list


'''
Save all images in: augmented_font_samples
'''


def save_images(image_list, name):
    for idx, image in enumerate(image_list):
        imageio.imwrite(f'augmented_font_samples/{name}_{str(idx)}.png', image)


def main():
    images_list = load_images()
    aug_elastic_list, aug_perspective_list, aug_affine = augment_images(images_list)
    save_images(aug_elastic_list, 'elastic_transform')
    save_images(aug_perspective_list, 'perspective_transform')
    save_images(aug_affine, 'affine_transform')


if __name__ == '__main__':
    main()
