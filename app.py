import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = YOLO("yolov10n.pt")  # your uploaded model
    return model

model = load_model()

# Title
st.title("YOLOv10 Object Detection")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to numpy array
    image_np = np.array(image)

    # Run detection
    results = model(image_np)

    # Plot results
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Detected Image", use_column_width=True)
