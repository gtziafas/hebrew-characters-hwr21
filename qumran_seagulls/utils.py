from .types import *

###  OpenCV utils #####
import numpy as np 
import cv2 


def show(img: array, legend: Maybe[str] = None):
    legend = 'unlabeled' if legend is None else legend
    cv2.imshow(legend, img)
    while 1:
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    cv2.destroyWindow(legend)


def crop_box(img: array, box: Box):
    h, w = img.shape[0:2]
    return img[h-box.top : h-box.bottom, box.left : box.right]


def test_contours(imgs: List[array]):
    for x in imgs:
        contours, _ = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)[1]
        mask = np.zeros((x.shape[0], x.shape[1], 3))
        mask = cv2.drawContours(mask, [cnt], 0, (0xff, 0, 0), -1)
        x0, y0, h, w = cv2.boundingRect(cnt)
        mask = cv2.rectangle(mask, (x0, y0), (x0+w, y0+h), (0, 0, 0xff), 1)
        canvas = cv2.merge((x, x, x))
        canvas = cv2.rectangle(canvas, (x0, y0), (x0+w, y0+h), (0, 0, 0xff), 1)
        show(np.hstack((canvas, mask)))