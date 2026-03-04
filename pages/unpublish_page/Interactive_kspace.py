import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize

def compute_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(1 + np.abs(fshift))
    phase = np.angle(fshift)
    return f, fshift, magnitude, phase

def reconstruct_image(fshift_filtered):
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_reconstructed = np.fft.ifft2(f_ishift)
    return np.abs(img_reconstructed)

def normalize_img(img):
    img = np.abs(img)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
        print("Image normalized")
    else:
        img = np.zeros_like(img)
        print("Image has no variation, returning zero array")
    img *= 255  # Scale to 0-255 for display
    img = img.astype(np.uint8)
    return img


st.set_page_config(layout="wide")
st.title("Interactive 2D Fourier Transform Demonstration")

# Parameters
img_size = 128
N = img_size # grid size
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)

# Placeholder image (phantom)
# phantom = np.exp(-((X**2 + Y**2) * 10))
phantom = data.shepp_logan_phantom()
phantom = resize(phantom, (img_size, img_size))\
    
# FFT of phantom = full k-space
kspace = np.fft.fftshift(np.fft.fft2(phantom))

# Streamlit interactive inputs
st.sidebar.subheader("Select k-space points")
num_points = st.sidebar.slider("Number of k-space points", 1, 10, 3)

selected_points = []
for i in range(num_points):
    u = st.sidebar.slider(f"u (kx) for point {i+1}", -N//2, N//2-1, 0)
    v = st.sidebar.slider(f"v (ky) for point {i+1}", -N//2, N//2-1, 0)
    selected_points.append((u, v))

# Create reconstruction from selected k-space points
recon = np.zeros((N, N), dtype=complex)
for (u, v) in selected_points:
    # map from slider index to FFT index
    ui = u + N//2
    vi = v + N//2
    recon[vi, ui] = kspace[vi, ui]
    st.write(f"Selected k-space point: ({u}, {v})")

# image_recon = np.abs(np.fft.ifft2(np.fft.ifftshift(recon)))
image_recon = reconstruct_image(recon)
# Plot layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original Image")
    st.image(phantom, clamp=True, use_container_width=True)
    st.write(selected_points)
with col2:
    st.subheader("K-space (selected points in red)")
    fig, ax = plt.subplots()
    ax.imshow(np.log1p(np.abs(kspace)), cmap="gray")
    for (u, v) in selected_points:
        ax.plot(u + N//2, v + N//2, "ro")
    st.pyplot(fig)

with col3:
    st.subheader("Reconstructed Image (from selected points)")
    st.image(normalize_img(image_recon), use_container_width=True)