import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. Setup
st.set_page_config(page_title="Pro AI Colorizer", layout="wide")
st.title("🎨 Ultra-Vibrant AI Colorizer")

PROTO = "colorization_deploy_v2.prototxt"
MODEL = "colorization_release_v2.caffemodel"
POINTS = "pts_in_hull.npy"

@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
    pts = np.load(POINTS).transpose().reshape(2, 313, 1, 1).astype("float32")
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full((1, 313), 2.606, "float32")]
    return net

net = load_model()

# 2. Upload
file = st.file_uploader("Upload B&W Photo", type=['jpg', 'png', 'jpeg'])

if file:
    img = Image.open(file).convert("RGB")
    img_np = np.array(img)
    
    # --- STEP 1: PRE-PROCESSING (FOR CONTRAST) ---
    # Convert to LAB to get the clean L channel
    img_scaled = img_np.astype("float32") / 255.0
    lab = cv2.cvtColor(img_scaled, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0] # This is the lightness
    
    # --- STEP 2: PREDICTION ---
    # Resize and shift by 50 (Crucial to prevent BLUE tint)
    resized_l = cv2.resize(l_chan, (224, 224))
    resized_l -= 50 
    
    blob = cv2.dnn.blobFromImage(resized_l)
    net.setInput(blob)
    ab_predicted = net.forward('class8_ab')[0, :, :, :].transpose((1, 2, 0))
    
    # Resize colors back to original image size
    ab_resized = cv2.resize(ab_predicted, (img_np.shape[1], img_np.shape[0]))
    
    # --- STEP 3: COLOR MERGING ---
    # Merge original L with predicted AB
    result_lab = np.concatenate((l_chan[:, :, np.newaxis], ab_resized), axis=2)
    res_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
    res_rgb = (np.clip(res_rgb, 0, 1) * 255).astype("uint8")

    # --- STEP 4: EXTREME COLOR BOOST ---
    # Force the colors to be vivid and colorful
    hsv = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2HSV).astype("float32")
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.2, 0, 255) # 2.2x Saturation boost
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255) # 1.1x Brightness boost
    final_image = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2RGB)

    # 3. Show Results
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original B&W", use_container_width=True)
    with col2:
        st.image(final_image, caption="AI Colorized (Enhanced)", use_container_width=True)
