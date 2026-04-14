"""
♻️ YOLO Trash Classification Web App (Improved)
================================================
Supports both:
  - Default COCO model (yolov8n.pt) for general detection
  - Fine-tuned TACO model (best.pt) for trash-specific detection

Deploy: streamlit run app.py
"""

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
from collections import Counter

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# Path to your fine-tuned model (update after training)
TACO_MODEL_PATH = "best.pt"
COCO_MODEL_PATH = "yolov8s.pt"  # Upgraded from yolov8n to yolov8s

# Check if TACO model exists
USE_TACO = os.path.exists(TACO_MODEL_PATH)

# ── Class mappings ──

# If using fine-tuned TACO model
TACO_WASTE_MAP = {
    "food_waste": {"type": "Biodegradable", "icon": "🍂", "color": "#2d8a4e"},
    "plastic": {"type": "Non-Biodegradable", "icon": "🔴", "color": "#c0392b"},
    "paper_cardboard": {"type": "Recyclable", "icon": "📦", "color": "#2980b9"},
    "glass": {"type": "Recyclable", "icon": "🫙", "color": "#27ae60"},
    "metal": {"type": "Recyclable", "icon": "🥫", "color": "#f39c12"},
    "other_waste": {"type": "Non-Biodegradable", "icon": "⚫", "color": "#7f8c8d"},
}

# If using COCO model (fallback — your original lists, expanded)
COCO_BIODEGRADABLE = [
    'banana', 'apple', 'broccoli', 'carrot', 'sandwich',
    'orange', 'pizza', 'donut', 'cake', 'hot dog',
]

COCO_NON_BIODEGRADABLE = [
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'laptop', 'mouse', 'keyboard', 'cell phone',
    'tv', 'remote', 'clock', 'vase', 'scissors', 'toothbrush',
]

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="YOLO Trash Classification",
    page_icon="♻️",
    layout="centered",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
    .waste-card {
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 4px solid;
    }
    .waste-card .label {
        font-weight: 600;
        font-size: 15px;
    }
    .waste-card .conf {
        font-size: 13px;
        opacity: 0.7;
    }
    .summary-box {
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        background: #f8f9fa;
    }
    .summary-box .count {
        font-size: 28px;
        font-weight: 700;
    }
    .summary-box .desc {
        font-size: 13px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────

st.title("♻️ YOLO Trash Classification")

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

if USE_TACO:
    st.caption("🎯 Using fine-tuned TACO model — optimized for real-world trash detection")
    model = load_model(TACO_MODEL_PATH)
else:
    st.caption("⚠️ Using default COCO model — for best results, train on TACO dataset")
    model = load_model(COCO_MODEL_PATH)

# ──────────────────────────────────────────────
# SIDEBAR SETTINGS
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Detection Settings")

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.35,
        step=0.05,
        help="Higher = fewer but more confident detections"
    )

    iou_threshold = st.slider(
        "IOU threshold (overlap removal)",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Lower = more aggressive duplicate removal"
    )

    use_tta = st.checkbox(
        "Enable test-time augmentation",
        value=False,
        help="Runs inference with flips/scales — slower but more accurate"
    )

    img_size = st.selectbox(
        "Input resolution",
        options=[320, 640, 1280],
        index=1,
        help="Higher = better for small objects but slower"
    )

    st.divider()
    st.caption(
        f"Model: `{'TACO (fine-tuned)' if USE_TACO else COCO_MODEL_PATH}`\n\n"
        f"Classes: `{len(TACO_WASTE_MAP) if USE_TACO else 80}`"
    )

# ──────────────────────────────────────────────
# CLASSIFICATION LOGIC
# ──────────────────────────────────────────────

def classify_detection(name: str, conf: float) -> dict:
    """Classify a detected object into waste categories."""
    name_lower = name.lower().strip()

    if USE_TACO:
        # TACO model — direct class mapping
        info = TACO_WASTE_MAP.get(name_lower, {
            "type": "Unknown",
            "icon": "❓",
            "color": "#95a5a6",
        })
        return {
            "name": name_lower,
            "confidence": conf,
            "waste_type": info["type"],
            "icon": info["icon"],
            "color": info["color"],
        }
    else:
        # COCO model — rule-based mapping
        if name_lower in COCO_BIODEGRADABLE:
            return {
                "name": name_lower,
                "confidence": conf,
                "waste_type": "Biodegradable",
                "icon": "🍂",
                "color": "#2d8a4e",
            }
        elif name_lower in COCO_NON_BIODEGRADABLE:
            return {
                "name": name_lower,
                "confidence": conf,
                "waste_type": "Non-Biodegradable",
                "icon": "🔴",
                "color": "#c0392b",
            }
        else:
            return {
                "name": name_lower,
                "confidence": conf,
                "waste_type": "Unknown",
                "icon": "❓",
                "color": "#95a5a6",
            }

# ──────────────────────────────────────────────
# IMAGE UPLOAD & DETECTION
# ──────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG, JPEG"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # Run detection
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        image.save(temp.name)

        with st.spinner("Detecting objects..."):
            results = model(
                temp.name,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=img_size,
                augment=use_tta,
            )

        # Clean up temp file
        os.unlink(temp.name)

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        # Classify all detections
        detections = []
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            name = results[0].names[cls]
            det = classify_detection(name, conf)
            detections.append(det)

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        # ── Summary statistics ──
        st.success(f"✅ Detected {len(detections)} object(s)")

        type_counts = Counter(d["waste_type"] for d in detections)
        avg_conf = sum(d["confidence"] for d in detections) / len(detections)

        cols = st.columns(len(type_counts) + 1)

        # Total count
        with cols[0]:
            st.markdown(f"""
            <div class="summary-box">
                <div class="count">{len(detections)}</div>
                <div class="desc">Total items</div>
            </div>
            """, unsafe_allow_html=True)

        # Per-type counts
        for i, (wtype, count) in enumerate(type_counts.most_common()):
            with cols[i + 1]:
                st.markdown(f"""
                <div class="summary-box">
                    <div class="count">{count}</div>
                    <div class="desc">{wtype}</div>
                </div>
                """, unsafe_allow_html=True)

        st.write("")

        # ── Detection list ──
        st.subheader("Detected objects")

        for det in detections:
            conf_pct = f"{det['confidence']:.0%}"
            st.markdown(f"""
            <div class="waste-card" style="border-color: {det['color']}; background: {det['color']}11;">
                <div class="label">{det['icon']} {det['name']} → {det['waste_type']}</div>
                <div class="conf">Confidence: {conf_pct}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Annotated image ──
        st.subheader("Detection overlay")
        st.image(results[0].plot(), caption="Annotated result", use_container_width=True)

        # ── Average confidence ──
        st.caption(f"Average confidence: {avg_conf:.0%} | Resolution: {img_size}px | "
                   f"TTA: {'on' if use_tta else 'off'}")

    else:
        st.warning("No objects detected. Try lowering the confidence threshold in the sidebar.")

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────

st.divider()
st.caption(
    "Built with [YOLOv8](https://docs.ultralytics.com) & "
    "[TACO Dataset](http://tacodataset.org) | "
    "Model: " + ("Fine-tuned TACO" if USE_TACO else "COCO (default)")
)
