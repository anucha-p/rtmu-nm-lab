import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import time

st.set_page_config(layout="wide", page_title="MRI Spatial Encoding & k-space Acquisition Demo")

# Helper functions
def fft2c(img):
    return fftshift(fft2(ifftshift(img)))

def ifft2c(kspace):
    return fftshift(ifft2(ifftshift(kspace)))

def normalize_img(img):
    img = np.abs(img)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img

img_size = 256
phantom = resize(shepp_logan_phantom(), (img_size, img_size))
phantom = normalize_img(phantom)
kspace_full = fft2c(phantom)
center = img_size // 2

st.title("MRI Spatial Encoding & k-space Acquisition Animation")

num_frames = st.slider("Number of animation frames", min_value=10, max_value=img_size, value=100)

# Placeholders for all parts
orig_img_placeholder = st.empty()
kspace_mag_plot_placeholder = st.empty()
kspace_img_placeholder = st.empty()
recon_img_placeholder = st.empty()
info_placeholder = st.empty()

# Initialize empty k-space accumulator
kspace_accum = np.zeros_like(kspace_full, dtype=complex)

# Calculate how many lines to acquire per frame (at least 1)
lines_per_frame = max(1, img_size // num_frames)

for frame in range(num_frames):
    # Calculate current k-space acquisition range (center out)
    half_lines = (frame * lines_per_frame) // 2
    start = max(center - half_lines, 0)
    end = min(center + half_lines + ((frame * lines_per_frame) % 2), img_size)

    # Update accumulated k-space with new lines acquired this frame
    kspace_accum[start:end, :] = kspace_full[start:end, :]

    # Reconstruct image from partial k-space
    img_recon = ifft2c(kspace_accum)
    img_recon_norm = normalize_img(img_recon)

    # -- Original image with spatial encoding overlays --

    fig_orig, ax_orig = plt.subplots(figsize=(6, 6))
    ax_orig.imshow(phantom, cmap='gray')

    # Frequency Encoding arrow (red, horizontal)
    ax_orig.arrow(20, img_size - 20, 80, 0, head_width=10, head_length=15, fc='red', ec='red', linewidth=2)
    ax_orig.text(60, img_size - 30, 'Frequency Encoding (Readout)', color='red', fontsize=10, ha='center')

    # Phase Encoding arrow (blue, vertical)
    ax_orig.arrow(20, img_size - 20, 0, -80, head_width=10, head_length=15, fc='blue', ec='blue', linewidth=2)
    ax_orig.text(5, img_size - 60, 'Phase Encoding', color='blue', fontsize=10, rotation=90, va='center')

    # Green dashed line for current phase encoding line (y-position)
    ax_orig.axhline(y=start, color='lime', linestyle='--', linewidth=2)
    ax_orig.axhline(y=end - 1, color='lime', linestyle='--', linewidth=2)

    ax_orig.set_axis_off()

    # -- k-space magnitude plot with acquisition indicator lines --

    fig_kspace, ax_kspace = plt.subplots(figsize=(6, 6))
    ax_kspace.imshow(np.log1p(np.abs(kspace_full)), cmap='gray')
    ax_kspace.axhline(y=start, color='red', linestyle='--', linewidth=1)
    ax_kspace.axhline(y=end - 1, color='red', linestyle='--', linewidth=1)
    ax_kspace.set_title("Full k-space magnitude with acquired region (red dashed lines)")
    ax_kspace.axis('off')

    # -- Display partial k-space magnitude --

    kspace_mag_norm = normalize_img(np.log1p(np.abs(kspace_accum)))

    # Update Streamlit UI elements
    orig_img_placeholder.subheader("Original Image with Spatial Encoding")
    orig_img_placeholder.pyplot(fig_orig)

    kspace_mag_plot_placeholder.subheader("Full k-space Magnitude with Acquisition Indicator")
    kspace_mag_plot_placeholder.pyplot(fig_kspace)

    kspace_img_placeholder.subheader(f"Partial k-space Data Acquired (lines {start} to {end - 1})")
    kspace_img_placeholder.image(kspace_mag_norm, clamp=True)

    recon_img_placeholder.subheader(f"Reconstructed Image from {end - start} k-space Lines")
    recon_img_placeholder.image(img_recon_norm, clamp=True)

    info_placeholder.markdown(
        f"Acquiring phase encoding lines from **{start}** to **{end - 1}** of k-space "
        f"(total **{end - start}** lines)."
    )

    time.sleep(0.05)

st.markdown("""
---
### Explanation

- The **original image** (phantom) shows the anatomy with overlaid gradient arrows:
  - **Red arrow:** Frequency encoding (x-axis) applied during signal readout.
  - **Blue arrow:** Phase encoding (y-axis) applied before readout to encode spatial position.
  - **Green dashed lines:** Indicate which phase encoding lines are currently being acquired (corresponding to k-space lines).

- The **full k-space magnitude** plot shows the entire spatial frequency domain with red dashed lines marking acquired lines.

- The **partial k-space** image displays the actual acquired data (non-zero lines) so far.

- The **reconstructed image** updates dynamically as more lines are acquired, improving image quality.

This visualization helps understand how MRI spatial encoding gradients link to k-space acquisition and image reconstruction.
""")