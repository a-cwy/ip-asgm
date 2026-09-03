import streamlit as st
import cv2 as cv
from morphological import run_pipeline;
from preprocessor import denoise, extract_roi

img = cv.imread("example.png")
img_denoised = denoise(img)
img_roi = extract_roi(img_denoised)
ratio, img_final, fruit_mask, blemish_mask = run_pipeline(img_roi)
overlay = img_final.copy()
overlay[blemish_mask > 0] = (0, 0, 255)

st.title("Fruit Blemish Detection & Classification")
st.image(img)
st.image(overlay)

