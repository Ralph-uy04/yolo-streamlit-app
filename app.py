import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile

# Define biodegradable & non-biodegradable categories
biodegradable_items = [
    'banana', 'apple', 'broccoli', 'carrot', 'sandwich',
    'orange', 'pizza', 'donut', 'cake', 'vegetable',
    'fruit', 'hot dog', 'bread', 'meat', 'fish', 'egg'
]

non_biodegradable_items = [
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'laptop', 'mouse', 'keyboard', 'cell phone',
    'tv', 'remote', 'clock', 'vase', 'scissors',
    'toothbrush'
]

# Page settings
st.set_page_config(page_title="YOLO Trash Classification", layout="centered")
st.title("♻️ YOLO Trash Classification Web App")

# Load model safely (NO manual download)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # auto-download handled by ultralytics

model = load_model()

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temp image
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp:
        image.save(temp.name)

        with st.spinner("🔍 Detecting objects..."):
            results = model(temp.name)

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        st.success("✅ Detection complete!")

        st.write("### 🎯 Detected Objects & Classification:")

        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            name = results[0].names[cls].lower()

            # Classification logic
            if name in biodegradable_items:
                classification = "Biodegradable ♻️"
            elif name in non_biodegradable_items:
                classification = "Non-Biodegradable 🚯"
            else:
                classification = "Unknown ❓"

            st.write(f"🔹 **{name}** ({conf:.2f}) → {classification}")

        # Show detection image
        st.image(results[0].plot(), caption="Detected Objects", use_column_width=True)

    else:
        st.warning("⚠️ No objects detected.")
