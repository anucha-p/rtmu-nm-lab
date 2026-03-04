import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.metrics import mean_squared_error, structural_similarity as ssim
from skimage.transform import resize

# --- Helper Functions ---
def compute_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(1 + np.abs(fshift))
    phase = np.angle(fshift)
    return f, fshift, magnitude, phase

def apply_filter(fshift, filter_type="lowpass", radius=30):
    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.uint8)

    if filter_type == "lowpass":
        mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1
    elif filter_type == "highpass":
        mask[:] = 1
        mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0
    else:
        mask[:] = 1  # no filtering

    fshift_filtered = fshift * mask
    return fshift_filtered, mask

def reconstruct_image(fshift_filtered):
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_reconstructed = np.fft.ifft2(f_ishift)
    return np.abs(img_reconstructed)

def normalize_img(img):
    img = np.abs(img)
    img -= img.min()
    img /= img.max() + 1e-8
    return img

# --- Streamlit App ---
st.title("🌀 2D Fourier Transform Demonstration")

# Select image
st.sidebar.header("Image Settings")
option = st.sidebar.selectbox("Choose an image", ["Camera", "Phantom", "Coins"])
if option == "Camera":
    img = color.rgb2gray(data.camera())
elif option == "Phantom":
    img = data.shepp_logan_phantom()
elif option == "Coins":
    img = data.coins()

# Resize for speed
img = resize(img, (256, 256), anti_aliasing=True)

# Compute FFT
f, fshift, magnitude, phase = compute_fft(img)

# Filter settings
st.sidebar.header("Filter Settings")
filter_type = st.sidebar.radio("Filter type", ["none", "lowpass", "highpass"])
radius = st.sidebar.slider("Filter radius", 5, 100, 30)

# Apply filter
fshift_filtered, mask = apply_filter(fshift, filter_type, radius)
reconstructed = reconstruct_image(fshift_filtered)

# Compute metrics
mse_value = mean_squared_error(img, reconstructed)
ssim_value = ssim(img, reconstructed, data_range=1.0)

# --- Display ---
col1, col2 = st.columns(2)
with col1:
    st.image(normalize_img(img), caption="Original Image", use_container_width=True, clamp=True)
    st.image(normalize_img(magnitude), caption="FFT Magnitude (log)", use_container_width=True, clamp=True)
with col2:
    st.image(normalize_img(phase), caption="FFT Phase", use_container_width=True, clamp=True)
    st.image(np.abs(reconstructed), caption="Reconstructed Image", use_container_width=True, clamp=True)

st.markdown("### Filter Mask in K-space")
st.image(mask*255, caption="Filter Mask", use_container_width=True, clamp=True)

st.markdown("### Metrics")
st.write(f"🔹 Mean Squared Error (MSE): {mse_value:.4f}")
st.write(f"🔹 Structural Similarity (SSIM): {ssim_value:.4f}")