import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os
from pathlib import Path

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
kvp_factor = 0.25 + ((kvp - 60) / (140 - 60)) * (1.75 - 0.25)  # map 60–140 → 0.25–1.75

# brightness_factor = mAs / 6.0  # map 0–1200 mA to brightness factor
# brightness_factor = 1.0 - min(mAs / 1000, 1.0) * 0.8  # maps 0–max_mAs → 1.0–0.2
brightness_factor = 1.25 - ((mAs - 2) / (10 - 2)) * (1.25 - 0.75)
# st.sidebar.markdown(f"**Brightness Factor:** `{brightness_factor:.2f}`")
# st.sidebar.markdown(f"**kVp Factor:** `{kvp_factor:.2f}`")


st.sidebar.markdown("---")

# --- Image Simulation ---
img = original_img.copy()
img = ImageEnhance.Contrast(img).enhance(kvp_factor)
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

with col1:
    st.subheader("📷 ภาพต้นฉบับ")
    st.image(original_img, use_container_width=True)

with col2:
    st.subheader("🔬 ภาพจำลอง")
    st.image(img, use_container_width=True)