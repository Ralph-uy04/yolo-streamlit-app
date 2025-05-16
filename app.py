import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import cv2

# Define biodegradable & non-biodegradable categories
biodegradable_items = [
    'banana', 'apple', 'broccoli', 'carrot', 'sandwich',
    'orange', 'book', 'pizza', 'donut', 'cake', 'vegetable',
    'fruit', 'hot dog', 'bread', 'meat', 'fish', 'egg'
]

non_biodegradable_items = [
    'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Page settings
st.set_page_config(page_title="YOLOv10 Trash Classification", layout="centered")
st.title("♻️ YOLOv10 Trash Classification Web App")

# Load model once
@st.cache_resource
def load_model():
    model_path = "yolov10n.pt"
    if not os.path.exists(model_path):
        from urllib.request import urlretrieve
        urlretrieve("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov10n.pt", model_path)
    return YOLO(model_path)

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        image.save(temp.name)
        results = model(temp.name)

    boxes = results[0].boxes
    detected_items = []

    if boxes:
        st.write("🎯 Detected Objects & Classification:")
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            name = results[0].names[cls].lower()

            # Classify object type
            if name in biodegradable_items:
                classification = "Biodegradable ♻️"
            elif name in non_biodegradable_items:
                classification = "Non-Biodegradable 🚯"
            else:
                classification = "Unknown ❓"

            detected_items.append({'label': name, 'type': classification})

            st.write(f"🔹 **{name}** ({conf:.2f}) → {classification}")

        st.image(results[0].plot(), caption="Detected Objects", use_column_width=True)
    else:
        st.warning("No objects detected.")
