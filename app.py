import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from PIL import Image
import joblib
from utils import ImagePreprocessor

model = joblib.load("mnist_svm_model.joblib")

st.set_page_config(page_title="MNIST MagicBox", layout="centered")
st.title("Draw and Predict MNIST Digit")

if "predictions" not in st.session_state:
    st.session_state.predictions = []

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Canvas")
    canvas = st_canvas(
        fill_color="#000000",
        stroke_color="#ffffff",
        stroke_width=8,
        background_color="#000000",
        width=300,
        height=300,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("Predict"):
        img = canvas.image_data.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        processor = ImagePreprocessor(0.1)
        imgs = processor.process_image(Image.fromarray(gray))

        results = []
        for img in imgs:
            prediction = model.predict(img.flatten().reshape(1, -1))[0]
            results.append((img, prediction))

        st.session_state.predictions = results

with col2:
    st.subheader("Predictions")
    if st.session_state.predictions:
        for i, (img, pred) in enumerate(st.session_state.predictions):
            st.image(img, caption=f"Image {i+1}", width=150)
            st.markdown(f"**Prediction:** `{pred}`")
