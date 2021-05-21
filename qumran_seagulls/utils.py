from .types import *

#####  OpenCV utils #####
import numpy as np 
import cv2 

MORPHOLOGY_MAP = {
    'erode': cv2.MORPH_ERODE,
    'dilate': cv2.MORPH_DILATE,
    'open': cv2.MORPH_OPEN,
    'close': cv2.MORPH_CLOSE
}


def show(img: array, legend: Maybe[str] = None):
    legend = 'unlabeled' if legend is None else legend
    cv2.imshow(legend, img)
    while 1:
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    cv2.destroyWindow(legend)


def show_many(imgs: List[array], legends: Maybe[List[str]] = None):
    assert len(imgs) == len(legends)
    legends = [l if l is not None else 'image' + str(i) for i, l in enumerate(legends)]
    print([f'{l}: {i.shape}'for l, i in zip(legends, imgs)])
    for i, l in zip(imgs, legends):
        cv2.imshow(l, i)
    while 1:
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    destroy()


def destroy():
    cv2.destroyAllWindows()


def crop_box(img: array, box: Box):
    return img[box.y : box.y + box.h, box.x : box.x + box.w]


# binarize and invert
def thresh_invert(img: array) -> array:
    _, thresh = cv2.threshold(img, 0, 0xff, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh 


def thresh_invert_many(imgs: List[array]) -> array:
    return list(map(thresh_invert, imgs))


# apply morphological kernel
def morphology(img: array, kernel: Tuple[int, int], morph: str = 'close', iterations: int=1) -> array:
    return cv2.morphologyEx(img, MORPHOLOGY_MAP[morph], kernel=np.ones(kernel, np.uint8), iterations=iterations)


# remove blobs of small area
def remove_blobs(img: array, area_thresh: int) -> array:
    # remove small blobs
    contours, _  = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) < area_thresh]
    mask = np.zeros_like(img)
    for c in contours:
        mask = cv2.drawContours(mask, [c], 0, 0xff, -1)
    img[mask==0xff] = 0
    return img

# thresh + invert + morphology + blobs
def denoise(img: array, kernel: Tuple[int, int], area_thresh: int) -> array:
    img = thresh_invert(img)
    img = morphology(img, kernel)
    return remove_blobs(img, area_thresh)


# pad image with zeros in the center of a desired resolution frame
def pad_with_frame(imgs: List[array], desired_shape: Tuple[int, int]) -> List[array]:
    H, W = desired_shape
    
    def _pad_with_frame(img: array) -> array:
        # construct a frame of desired resolution
        frame = np.zeros((H, W))

        # paste image in the center of the frame
        startx, starty = (H - img.shape[0]) // 2, (W - img.shape[1]) // 2
        frame[startx : startx + img.shape[0], starty :  starty + img.shape[1]] = img
        return frame

    return list(map(_pad_with_frame, imgs))


def filter_large(desired_shape: Tuple[int, int]) -> Callable[[List[array]], List[array]]:
    H, W = desired_shape

    def _filter_large(imgs: List[array]) -> List[array]: 
        # identify images larger than desired resolution
        large_idces = [idx for idx, i in enumerate(imgs) if i.shape[0] > H or i.shape[1] > W]

        # crop tight boxes for that imges
        cropped_imgs = crop_boxes_dynamic([imgs[idx] for idx in large_idces])

        # return all thresholded and inverted and large properly replaced
        return [thresh_invert(img) if i not in large_idces else cropped_imgs[large_idces.index(i)] for i, img in enumerate(imgs)]
    
    return _filter_large


def crop_boxes_dynamic(imgs: List[array]) -> List[array]:
    # threshold and invert
    imgs = thresh_invert_many(imgs)
    
    # find contours for each image
    contours = [cv2.findContours(i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for i in imgs]
    
    # sort by area and keep largest
    contours = [sorted(conts, key=lambda c: cv2.contourArea(c), reverse=True)[0] for conts in contours]
    
    # compute largests contour bounding box
    boxes = list(map(cv2.boundingRect, contours))

    # crop tight box
    return [crop_box(i, Box(*b)) for i, b in zip(imgs, boxes)]


def crop_boxes_fixed(desired_shape: Tuple[int, int]) -> Callable[[List[array]], List[array]]:
    H, W = desired_shape

    def _crop_boxes_fixed(imgs: List[array]) -> List[array]:
        # threshold and invert
        imgs = thresh_invert_many(imgs)
        
        # identify images larger than desired resolution
        large_idces = [idx for idx, i in enumerate(imgs) if i.shape[0] > H or i.shape[1] > W]

        # find contours for each large image
        contours = [cv2.findContours(imgs[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for i in large_idces]
        
        # sort by area and keep largest
        contours = [sorted(conts, key=lambda c: cv2.contourArea(c), reverse=True)[0] for conts in contours]
        
        # compute largests contour center of gravity
        moments = [cv2.moments(c) for c in contours]
        centers = [(int(M['m10']/M['m00']), int(M['m01']/M['m00'])) for M in moments]

        # fix images in desired resolution
        for idx, center in zip(large_idces, centers):
            image = imgs[idx]
            height, width = image.shape 
            cx = max(0, center[0] - min(W, width) // 2)
            cy = max(0, center[1] - min(H, height) // 2)
            box = Box(cx, cy, min(W, width), min(H, height))
            imgs[idx] = crop_box(image, box)

        return pad_with_frame(imgs, (H, W))

    return _crop_boxes_fixed