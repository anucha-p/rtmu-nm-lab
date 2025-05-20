import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os
from pathlib import Path
import io

# --- CONFIG ---
st.set_page_config(page_title="X-ray Simulator", page_icon="☢️", layout="centered")
# INPUT_IMAGE = "chest_xray.jpg"
# OUTPUT_DIR = "output"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent

# --- Load image ---
# image_base64 = img_to_bytes(BASE_DIR / 'images/CTsimulator/desktop.png')
original_img = Image.open(BASE_DIR / 'images/xray/chest_xray.jpeg').convert("RGB")

# --- Sidebar controls ---
st.sidebar.header("🔧 ปรับค่าการตั้งค่าเครื่อง X-ray")
default_kvp = 100
default_mas = 6

# Initialize session state for sliders if not already set
if 'kvp' not in st.session_state:
    st.session_state['kvp'] = default_kvp
if 'mAs' not in st.session_state:
    st.session_state['mAs'] = default_mas

# Add reset button
if st.sidebar.button("รีเซ็ตค่าเริ่มต้น"):
    st.session_state['kvp'] = default_kvp
    st.session_state['mAs'] = default_mas

# Use only the key argument to let Streamlit manage the value via session state
kvp = st.sidebar.slider(
    "kVp", min_value=60, max_value=140, step=10, key='kvp'
)
mAs = st.sidebar.slider(
    "mAs", min_value=2, max_value=10, step=1, key='mAs'
)
# st.sidebar.markdown(f"**Exposure Time:** `{default_time_s}`")
st.sidebar.markdown("---")

# คำนวณค่าทางเทคนิค

# Map kVp (60–140) → contrast_factor (1.3 → 0.7)
kvp_min, kvp_max = 60, 140
# contrast_factor = 0.25 + ((kvp - kvp_min) / (kvp_max - kvp_min)) * (1.75 - 0.25)  # map 60–140 → 0.25–1.75
contrast_factor = 2.0 - ((kvp - kvp_min) / (kvp_max - kvp_min)) * (2.0 - 0.01)

# brightness_factor = mAs / 6.0  # map 0–1200 mA to brightness factor
# brightness_factor = 1.0 - min(mAs / 1000, 1.0) * 0.8  # maps 0–max_mAs → 1.0–0.2
mas_min, mas_max = 2, 10
brightness_factor = 1.25 - ((mAs - mas_min) / (mas_max - mas_min)) * (1.25 - 0.75)
st.sidebar.markdown(f"**Brightness Factor:** `{brightness_factor:.2f}`")
st.sidebar.markdown(f"**contrast_factor:** `{contrast_factor:.2f}`")


st.sidebar.markdown("---")

# --- Image Simulation ---
img = original_img.copy()
img = ImageEnhance.Contrast(img).enhance(contrast_factor)

# # Adjust brightness based on contrast_factor
# if contrast_factor > 1.0:
#     # Higher contrast, lower brightness slightly
#     img = ImageEnhance.Brightness(img).enhance(0.95)
# elif contrast_factor < 1.0:
#     # Lower contrast, increase brightness slightly
#     img = ImageEnhance.Brightness(img).enhance(1.05)

# Normalize image before enhancing brightness
# img_array = np.array(img).astype(np.float32)
# img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255
# img = Image.fromarray(img_array.astype(np.uint8))

img = ImageEnhance.Brightness(img).enhance(brightness_factor)

# Add noise if mAs ต่ำ
img_array = np.array(img).astype(np.int16)
if mAs < 6:
    noise = np.random.normal(loc=0, scale=21, size=img_array.shape).astype(img_array.dtype)
    img_array += noise
img_array = np.clip(img_array, 0, 255).astype(np.uint8)
img = Image.fromarray(img_array)

# Blur if time_s สูง
# if blur_radius > 0:
#     img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))


# --- Display ---
st.title("🩻 Radiographic Image Simulator")
st.markdown("ปรับค่าการถ่ายภาพรังสีเพื่อดูผลที่เกิดกับภาพ")

col1, col2 = st.columns(2)

import matplotlib.pyplot as plt

with col1:
    st.subheader("📷 ภาพต้นฉบับ")
    st.image(original_img, use_container_width=True)

with col2:
    st.subheader("🔬 ภาพจำลอง")
    st.image(img, use_container_width=True)

    # # Add color bar to demonstrate contrast
    # st.markdown("**Color Bar (Pixel Intensity):**")
    # fig, ax = plt.subplots(figsize=(5, 0.5))
    # fig.subplots_adjust(bottom=0.5)

    # cmap = plt.get_cmap('gray')
    # norm = plt.Normalize(vmin=0, vmax=255)
    # cb1 = plt.colorbar(
    #     plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    #     cax=ax, orientation='horizontal'
    # )
    # cb1.set_label('Pixel Intensity')
    # buf = io.BytesIO()
    # plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
    # plt.close(fig)
    # st.image(buf.getvalue())