import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Page settings
st.set_page_config(page_title="YOLOv10 Object Detection", layout="centered")
st.title("🔍 YOLOv10 Object Detection Web App")

# Load YOLOv10n model once (cached)
@st.cache_resource
def load_model():
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
uploaded_file = st.file_uploader("📁 Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    # Save image to a temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        image.save(temp.name)
        results = model(temp.name)

    # Process detection results
    result = results[0]
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        st.subheader("🎯 Detected Objects:")
        for box in boxes:
            cls_id = int(box.cls)
            confidence = float(box.conf)
            label = result.names[cls_id]
            st.write(f"🔹 **{label}** with confidence **{confidence:.2f}**")

        # Show detection results with boxes
        st.image(result.plot(), caption="📦 Detected Objects", use_column_width=True)
    else:
        st.warning("🚫 No objects detected in the image.")
