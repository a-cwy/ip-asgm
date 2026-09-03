import numpy as np
import cv2 as cv

def remove_black(img):
  # Masking to remove black background
  lower_black = np.array([0, 0, 0], dtype="uint8")
  upper_black = np.array([10, 10, 10], dtype="uint8")

  black_mask = cv.inRange(img, lower_black, upper_black)

  img[black_mask > 0] = [255, 255, 255]

  return img

def remove_noise(img):
  median_filtered = cv.medianBlur(img, 5)

  return median_filtered

def denoise(img):
  img = remove_black(img)
  img = remove_noise(img)
  return img

def extract_roi(img):
  # OTSU thresholding
  img_sat = cv.cvtColor(img, cv.COLOR_RGB2HSV)[:, :, 1]
  ret, thresh = cv.threshold(img_sat, 0, 255, cv.THRESH_OTSU)

  # Noise removal
  kernel = np.ones((3,3),np.uint8)
  opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 2)

  # ROI
  eroded = cv.erode(opened, kernel, iterations = 3)
  ret, sure_fg = cv.threshold(eroded, 0.5 * eroded.max(), 255, 0)

  # RONI
  sure_bg = cv.dilate(opened, kernel, iterations = 3)

  # Finding unknown region
  sure_fg = np.uint8(sure_fg)
  unknown = cv.subtract(sure_bg, sure_fg)

  # Markers
  ret, markers = cv.connectedComponents(sure_fg)
  markers = markers + 1
  markers[unknown == 255] = 0

  markers = cv.watershed(img, markers)
  img_ws = img.copy()
  img_ws[markers == 1] = [0, 0, 0]

  y, x = np.where(markers > 1)
  xy_min = (np.min(x), np.min(y))
  xy_max = (np.max(x), np.max(y))
  bbox = [xy_min, xy_max]

  img_ws_bbox = img_ws.copy()
  cv.rectangle(img_ws_bbox, bbox[0], bbox[1], (0, 255, 0), 2)

  img_roi = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0], :]
  img_roi = cv.resize(img_roi, (128, 128), interpolation=cv.INTER_AREA)

  return img_roi