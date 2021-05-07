from ..types import *
from ..utils import *

import cv2
import numpy as np 
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import simps


MORPH_KERNEL = (2, 2)
BLOB_CUTTOF_THRESH = 100
BINARY_COUNT_THRESH = 10
PEAKS_MERGE_THRESH = 10
WIDTH_SCALER = 0.7


# step 1
def denoise(img: array, kernel: Tuple[int, int], area_thresh: int) -> array:
    # threshold and inverse
    _, thresh = cv2.threshold(img, 0, 0xff, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # morphology
    kernel = np.ones(kernel, np.uint8)
    thresh =  cv2.erode(thresh, kernel, iterations=1)

    # remove small blobs
    contours, _  = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) < area_thresh]
    mask = np.zeros_like(thresh)
    for c in contours:
        mask = cv2.drawContours(mask, [c], 0, 0xff, -1)
    thresh[mask==0xff] = 0
    return thresh


# identify consecutive 0s in a 1-d array
def zero_runs(arr: array) -> List[List[int]]:
    iszero = np.concatenate(([0], np.equal(arr, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return [list(range(r[0], r[1])) for r in ranges]


# merge 0 peaks to corresponding adjacent peaks
# @todo: when have time, optimize this crappy code
def merge_zero_peaks(peaks: array, widths: array, ranges: List[List[int]]) -> Tuple[array, ...]:
    new_peaks, new_widhts = [], []

    i = 0
    while i < len(peaks):
        _size = 1
        _peak = peaks[i]
        _width = widths[i]
        exception = False 

        for r in ranges:
            if i in r:
                if len(r) > 1:
                    _size = len(r)
                    _peak = sum([peaks[j] for j in r]) / _size 
                    _width = sum([widths[j] for j in r])
                    break 

                else:
                    dist_left  = peaks[i] - peaks[i-1] if i>0 else 1e09
                    dist_right = peaks[i+1] - peaks[i] if i+1 < len(peaks) else 1e09

                    if dist_left < dist_right:
                        new_peaks[-1] = (new_peaks[-1] + peaks[i]) // 2
                        new_widhts[-1] += widths[i]
                        exception = True
                        break

                    else:
                        _size = 2
                        _peak = (peaks[i] + peaks[i+1]) // 2
                        _width = widths[i] + widths[i+1]
                        break

        i += _size
        if not exception:
            new_peaks.append(_peak)
            new_widhts.append(_width)

    return array(new_peaks).astype(int), array(new_widhts).astype(int)


# merge close peaks recursively to respect a total width threshold
def recursive_merge(peaks: array, widths: array, w_thresh: int) -> Tuple[List[int], List[int]]:
    # base case
    if widths.sum() <= w_thresh:
        _peak = int(peaks.mean())
        _width = widths.sum()
        return [_peak], [_width]
    
    # split according to max dist
    split_id = np.diff(peaks).argmax() + 1
    peaks1, widths1 = recursive_merge(peaks[:split_id], widths[:split_id], w_thresh)
    peaks2, widths2 = recursive_merge(peaks[split_id:], widths[split_id:], w_thresh)
    return peaks1 + peaks2, widths1 + widths2


def merge_close_peaks(peaks: array, widths: array, dist_thresh: int, stepsize: int=1) -> Tuple[array, ...]:
    new_peaks = peaks.tolist()
    new_widths = widths.tolist()

    # get indices of peaks that are closer than threshold
    close_ids = np.where(np.diff(peaks) < dist_thresh)[0]
    # split them into consecutive
    close_ids = np.split(close_ids, np.where(np.diff(close_ids) != stepsize)[0]+1)
    if not len(close_ids[0]):
        return peaks, widths
    close_ids = [np.append(ids, ids.max() + 1) for ids in close_ids]

    # replace with merged peaks
    offset = 0
    for ids in close_ids:
        ps, ws = recursive_merge(peaks[ids], widths[ids],  widths.max())
        new_peaks[ids[0]-offset : ids[-1]+1-offset] = ps
        new_widths[ids[0]-offset : ids[-1]+1-offset] = ws
        offset += len(peaks[ids]) - len(ps) 

    return array(new_peaks).astype(int), array(new_widths).astype(int)


# segment characters from a given image of a line
def segment_chars_from_line(line: array,
                            kernel: Tuple[int, int] = MORPH_KERNEL,
                            area_thresh: int = BLOB_CUTTOF_THRESH,
                            count_thresh: int = BINARY_COUNT_THRESH,
                            dist_thresh: int = PEAKS_MERGE_THRESH,
                            kappa: float = WIDTH_SCALER
                            ) -> List[array]:
    clean = denoise(line, kernel, area_thresh)

    # count written pixels vertically
    vertical_count = np.where(clean > 0, 1, 0).sum(axis=0)

    # convert to binary respecting a small threshold
    binary_count = np.where(vertical_count > count_thresh, 1, 0)
    
    # get number and width of peaks from binary count
    peaks, _ = find_peaks(binary_count)
    widths = peak_widths(binary_count, peaks)[0]
    w_thresh = widths.mean() * kappa

    # merge close peaks
    peaks, widths = merge_close_peaks(peaks, widths, dist_thresh)

    # compute areas
    areas = array([simps(binary_count[p - w//2 : p + w//2 + 1]) for w,p in zip(widths, peaks)])

    # flag each peak as over/wrong (0) according to average width
    #flags = np.where(areas > 75, 1, 0)
    flags = np.where(widths > w_thresh, 1, 0)
    
    # get consecutive over/wrong peaks
    ranges = zero_runs(flags)

    # run adaptive pixel counting to merge over/wrong peaks
    peaks, widths = merge_zero_peaks(peaks, widths, ranges)

    # get proposed segments while removing intersecting one
    spans = [(int(p-w/2), int(p+w/2)) for p, w in zip(peaks, widths)]
    segments = [(s[1] + spans[i+1][0]) // 2 for i, s in enumerate(spans[:-1])]
    segments_fil = [s for s in segments if np.count_nonzero(clean[:, s][:-1] < clean[:, s][1:]) < 2]
    # visualize
    mask = clean.copy()
    visualize({'line': line, 'clean': clean, 'mask': mask, 'vertical': vertical_count, 
               'segments': segments_fil, 'segments_cut': [s for s in segments if s not in segments_fil],
               'binary': binary_count, 'peaks0': [p for i,p in enumerate(peaks) if flags[i]==0], 
               'peaks1': [p for i,p in enumerate(peaks) if flags[i]==1]})

    # crop original image to segments
    ...
    return 1


def visualize(stuff: Dict[str, Any]):
    import matplotlib.pyplot as plt 
    fig, (ax1, ax3, ax4) = plt.subplots(3, sharex=True)
    fig.suptitle('Character segmentation - adaptive pixel count')
    ax1.imshow(stuff['line'])
    ax3.imshow(stuff['mask'])
    ax3.vlines(stuff['segments'], 0, 60, color="C1", linestyle='solid', label='true')
    ax3.vlines(stuff['segments_cut'], 0, 60, color="C1", linestyle='dotted', label='false')
    ax4.plot(stuff['vertical'], 'b-', label='vertical')
    ax4.plot(stuff['binary'] * 20, 'y--', label='binary')
    ax4.plot(stuff['peaks0'], [20] * len(stuff['peaks0']), 'rx', label='0-peaks')
    ax4.plot(stuff['peaks1'], [20] * len(stuff['peaks1']), 'gx', label='1-peaks')
    ax4.vlines(stuff['segments'], 0, 60, color="C1", linestyle='solid', label='true')
    ax4.vlines(stuff['segments_cut'], 0, 60, color="C1", linestyle='dotted', label='false')
    ax4.grid(True)
    plt.legend()
    plt.show()