import numpy as np
import cv2 as cv

MORPH_SHAPES = {'ellipse': cv.MORPH_ELLIPSE, 'rect': cv.MORPH_RECT, 'cross': cv.MORPH_CROSS}

def detect_blemishes(img, fruit_mask, k=15, shape='ellipse', operator='blackhat', clean_k=3):
    """Detect blemish pixels inside `fruit_mask` using a morphological operator.
    operator: 'blackhat' (dark spots), 'tophat' (bright spots), or 'gradient' (edges).
    Returns a single-channel 0/255 mask restricted to the fruit region.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    fruit_vals = gray[fruit_mask > 0]
    fill_value = int(np.median(fruit_vals)) if fruit_vals.size else 0
    gray_masked = gray.copy()
    gray_masked[fruit_mask == 0] = fill_value

    kernel = cv.getStructuringElement(MORPH_SHAPES[shape], (k, k))
    if operator == 'blackhat':
        morphed = cv.morphologyEx(gray_masked, cv.MORPH_BLACKHAT, kernel)
    elif operator == 'tophat':
        morphed = cv.morphologyEx(gray_masked, cv.MORPH_TOPHAT, kernel)
    elif operator == 'gradient':
        morphed = cv.morphologyEx(gray_masked, cv.MORPH_GRADIENT, kernel)
    else:
        raise ValueError(f'Unknown operator: {operator}')

    if not np.any(morphed):
        return np.zeros_like(morphed)
    _, blemish = cv.threshold(morphed, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    clean_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (clean_k, clean_k))
    blemish = cv.morphologyEx(blemish, cv.MORPH_OPEN, clean_kernel)
    blemish = cv.bitwise_and(blemish, blemish, mask=fruit_mask)
    return blemish

def blemish_ratio(fruit_mask, blemish_mask):
    """Fraction of the fruit's surface area flagged as a blemish."""
    fruit_px = int(np.count_nonzero(fruit_mask))
    if fruit_px == 0:
        return 0.0
    blemish_px = int(np.count_nonzero(blemish_mask))
    return blemish_px / fruit_px


def classify(ratio, threshold):
    return 'rotten' if ratio > threshold else 'fresh'


def run_pipeline(img, k=15, shape='ellipse', operator='blackhat'):
    """Full per-image pipeline. Returns (ratio, img, fruit_mask, blemish_mask) or
    (None, None, None, None) if the image could not be read.
    """
    fruit_mask = segment_fruit(img)
    blemish_mask = detect_blemishes(img, fruit_mask, k=k, shape=shape, operator=operator)
    ratio = blemish_ratio(fruit_mask, blemish_mask)
    return ratio, img, fruit_mask, blemish_mask

def _fix_polarity(thresh):
    """Otsu thresholding does not know which side is the fruit. Since the fruit is
    always roughly centred and the background always touches the image border, flip
    the mask if the border pixels are mostly foreground.
    """
    border = np.zeros_like(thresh, dtype=bool)
    bw = 5
    border[:bw, :] = border[-bw:, :] = border[:, :bw] = border[:, -bw:] = True
    if (thresh[border] == 255).mean() > 0.5:
        thresh = cv.bitwise_not(thresh)
    return thresh

def segment_fruit(img, close_k=15, open_k=9):
    """Segment the fruit from the background. Returns a single-channel 0/255 mask."""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    _, thresh = cv.threshold(s, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresh = _fix_polarity(thresh)

    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_k, close_k))
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (open_k, open_k))
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, close_kernel)
    opened = cv.morphologyEx(closed, cv.MORPH_OPEN, open_kernel)

    contours, _ = cv.findContours(opened, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(opened)
    if contours:
        largest = max(contours, key=cv.contourArea)
        cv.drawContours(mask, [largest], -1, 255, thickness=cv.FILLED)
    return mask

def compute_color_score(img, fruit_mask, sat_drop_thresh=60, val_drop_thresh=60):
    """Fraction of the fruit's pixels that look 'unhealthy' in HSV -- dull/brownish
    relative to the fruit's own healthy-region colour. Catches diffuse spoilage
    (browning, dulling) that has no distinct blob, so Black-Hat morphology misses it.
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    fruit_px = fruit_mask > 0
    if not np.any(fruit_px):
        return 0.0

    # Use the brightest/most-saturated half of the fruit as a proxy for "healthy skin"
    healthy_ref_s = np.percentile(s[fruit_px], 75)
    healthy_ref_v = np.percentile(v[fruit_px], 75)

    dull_mask = (s.astype(int) < healthy_ref_s - sat_drop_thresh) | \
                (v.astype(int) < healthy_ref_v - val_drop_thresh)
    dull_mask &= fruit_px

    return float(np.count_nonzero(dull_mask)) / float(np.count_nonzero(fruit_px))

