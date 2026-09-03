import streamlit as st
import cv2 as cv
import numpy as np
import PIL
from morphological import run_pipeline;
from preprocessor import denoise, extract_roi

img_display = []
overlay = []
ratio = 0.0
THRESHOLD = 0.1030

def classify():
    global img_display
    global overlay
    global ratio

    img_colorcorrected = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_denoised = denoise(img)
    img_roi = extract_roi(img_denoised)
    ratio, img_final, fruit_mask, blemish_mask = run_pipeline(img_roi)
    overlay = img_final.copy()
    overlay[blemish_mask > 0] = (0, 0, 255)
    img_display = cv.resize(img_colorcorrected, (256, 256), interpolation=cv.INTER_AREA)

st.title("Fruit Blemish Detection & Classification")

img_uploaded = st.file_uploader(label = "Input Image", type = ["jpg", "png", "webp"], accept_multiple_files = False)
if img_uploaded is None:
    img = cv.imread("example.png")
else:
    file_bytes = np.asarray(bytearray(img_uploaded.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, 1)
classify()

st.image(
    [img_display, overlay],
    caption = ["Original Image", "After processing"]    
)
st.text(f"Class: {"Fresh" if ratio < THRESHOLD else "Rotten"}")
st.text(f"Blemish Ratio {ratio:.4f}")