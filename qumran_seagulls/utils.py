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
    x0, y0, w, h = box
    return img[y0 : y0+h, x0 : x0+w]


def denoise(img: array, kernel: Tuple[int, int], area_thresh: int) -> array:
    # threshold and inverse
    _, thresh = cv2.threshold(img, 0, 0xff, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # morphology
    kernel = np.ones(kernel, np.uint8)
    #thresh =  cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    # remove small blobs
    contours, _  = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) < area_thresh]
    mask = np.zeros_like(thresh)
    for c in contours:
        mask = cv2.drawContours(mask, [c], 0, 0xff, -1)
    thresh[mask==0xff] = 0
    return thresh


def tighten_boxes(imgs: List[array]):
    # threshold and invert
    imgs_thr = [cv2.threshold(i, 0, 0xff, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] for i in imgs]
    
    # find contours for each image
    contours = [cv2.findContours(i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for i in imgs_thr]
    
    # sort by area and keep largest
    contours_max = [sorted(conts, key=lambda c: cv2.contourArea(c), reverse=True)[0] for conts in contours]
    
    # compute largest contour center of gravity and bounding box
    #moments = [cv2.moments(c) for c in contours_max]
    #centers = [(int(M['m10']/M['m00']), int(M['m01']/M['m00'])) for M in moments]
    boxes = [cv2.boundingRect(c) for c in contours_max]

    # crop tight box
    return [crop_box(i, box) for i, box in zip(imgs_thr, boxes)]



