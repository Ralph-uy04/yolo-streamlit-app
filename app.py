import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Page settings
st.set_page_config(page_title="YOLOv10 Object Detection", layout="centered")
st.title("🔍 YOLOv10 Object Detection Web App")

# Load model once
@st.cache_resource
def load_model():
    # Auto-download YOLOv10n if not present
    model_path = "yolov10n.pt"
    if not os.path.exists(model_path):
        from urllib.request import urlretrieve
        urlretrieve(
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov10n.pt",
            model_path
        )
    return YOLO(model_path)

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to temp file for YOLO
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        image.save(temp.name)
        results = model(temp.name)

    boxes = results[0].boxes
    if boxes:
        st.write("🎯 Detected Objects:")
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            name = results[0].names[cls]
            st.write(f"🔹 {name} ({conf:.2f})")

        # Show image with boxes
        st.image(results[0].plot(), caption="Detected Objects", use_column_width=True)
    else:
        st.warning("No objects detected.")
