import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from model import OilModel, DebrisModel

# -------------------------
# PAGE SETUP
# -------------------------
st.set_page_config(page_title="Water Pollution AI", layout="centered")

st.title("🌊 Water Pollution Detection System")
st.write("AI-based detection of Oil Spills and Floating Debris")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD MODELS
# -------------------------
oil_model = OilModel().to(device)
oil_model.load_state_dict(torch.load("oil_model.pth", map_location=device))
oil_model.eval()

debris_model = DebrisModel(num_classes=6).to(device)
debris_model.load_state_dict(torch.load("debris_model.pth", map_location=device))
debris_model.eval()

oil_classes = ["Non Oil Spill", "Oil Spill"]
debris_classes = ["cardboard","glass","metal","paper","plastic","trash"]

# -------------------------
# UPLOAD IMAGE
# -------------------------
uploaded = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------
    # PREPROCESS (MATCH TRAINING)
    # -------------------------
    img = np.array(image)

    # Resize to 256
    img = cv2.resize(img, (256,256)) / 255.0

    # Center crop to 224
    start = 16
    end = 240
    img = img[start:end, start:end]

    # Normalize (VERY IMPORTANT)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # Convert to tensor
    img = np.transpose(img, (2,0,1))
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    st.subheader("🔍 Analysis Results")

    # -------------------------
    # OIL PREDICTION
    # -------------------------
    oil_pred = oil_model(img_tensor)
    oil_probs = torch.softmax(oil_pred, dim=1)
    oil_conf, oil_idx = torch.max(oil_probs, 1)

    oil_conf = oil_conf.item()
    oil_idx = oil_idx.item()

    # -------------------------
    # DEBRIS PREDICTION
    # -------------------------
    debris_pred = debris_model(img_tensor)
    debris_probs = torch.softmax(debris_pred, dim=1)
    debris_conf, debris_idx = torch.max(debris_probs, 1)

    debris_conf = debris_conf.item()
    debris_idx = debris_idx.item()

    # -------------------------
    # FINAL DECISION LOGIC
    # -------------------------

    # Oil detection (strict)
    if oil_conf > 0.90 and oil_idx == 1:
        st.error(f"🛢 Oil Spill Detected ({oil_conf:.2f})")
        st.info("🗑 Debris Analysis Skipped")

    else:
        if oil_conf > 0.7:
            st.warning(f"🛢 Possible Oil Spill ({oil_conf:.2f})")
        else:
            st.success("🌊 No Oil Spill Detected")

        # Debris detection
        if debris_conf > 0.6:
            st.info(f"🗑 Debris Type: {debris_classes[debris_idx]} ({debris_conf:.2f})")
        else:
            st.warning("🗑 Debris Detection Uncertain")

    # -------------------------
    # OPTIONAL DEBUG INFO
    # -------------------------
    with st.expander("📊 Confidence Details"):
        st.write("Oil Probabilities:", oil_probs.detach().cpu().numpy())
        st.write("Debris Probabilities:", debris_probs.detach().cpu().numpy())