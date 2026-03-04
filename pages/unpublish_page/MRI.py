import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage.transform import resize
from numpy.fft import fftshift, ifftshift, fft2, ifft2

st.set_page_config(layout="wide", page_title="MRI Reconstruction Teaching Tool")

# -----------------------------
# Helper Functions
# -----------------------------
def fft2c(img):
    return fftshift(fft2(ifftshift(img)))

def ifft2c(kspace):
    return fftshift(ifft2(ifftshift(kspace)))

def generate_coil_sensitivities(n_coils, img_shape):
    """Simulate simple coil sensitivity maps."""
    x = np.linspace(-1, 1, img_shape[0])
    y = np.linspace(-1, 1, img_shape[1])
    X, Y = np.meshgrid(x, y)
    sensitivities = []
    for c in range(n_coils):
        angle = 2 * np.pi * c / n_coils
        sens = np.exp(-((X - 0.3*np.cos(angle))**2 + (Y - 0.3*np.sin(angle))**2) / 0.2)
        sens = sens * np.exp(1j * angle)
        sensitivities.append(sens)
    return np.array(sensitivities)

def simulate_coil_sensitivities(shape, n_coils, smoothness=0.3):
    """
    Simulate smooth coil sensitivity maps arranged evenly around the FOV.
    Each coil has a 2D Gaussian profile centered at different angles on a circle.
    smoothness controls Gaussian std deviation relative to image size.
    """
    x = np.linspace(-1, 1, shape[1])
    y = np.linspace(-1, 1, shape[0])
    X, Y = np.meshgrid(x, y)
    coords = np.stack([X, Y], axis=-1)

    coil_maps = np.zeros((n_coils, *shape), dtype=np.complex64)
    radius = 0.7
    sigma = smoothness

    for c in range(n_coils):
        angle = 2 * np.pi * c / n_coils
        cx, cy = radius * np.cos(angle), radius * np.sin(angle)
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mag = np.exp(- (dist ** 2) / (2 * sigma ** 2))
        phase = np.exp(1j * 2 * np.pi * (X * cx + Y * cy))  # small spatial phase variation
        coil_maps[c] = mag * phase

    return coil_maps

def sos_reconstruction(coil_images):
    return np.sqrt(np.sum(np.abs(coil_images)**2, axis=0))

def sense_reconstruction(kspace_coils, coil_sensitivities):
    """
    Basic SENSE reconstruction solving pixel-wise linear system:
    For each pixel, solve s = S * m
    where s = vector of coil pixel values,
          S = coil sensitivity matrix,
          m = underlying pixel intensity (scalar).
    Here we solve for m by linear least squares.
    
    This is a simplified example ignoring noise covariance and regularization.
    """
    n_coils, nx, ny = kspace_coils.shape
    # Reconstruct each coil image (complex)
    coil_imgs = np.zeros_like(kspace_coils, dtype=np.complex64)
    for c in range(n_coils):
        coil_imgs[c] = ifft2c(kspace_coils[c])

    # Prepare output image
    recon = np.zeros((nx, ny), dtype=np.complex64)

    # For each pixel, solve linear system
    for ix in range(nx):
        for iy in range(ny):
            S = coil_sensitivities[:, ix, iy].reshape((n_coils, 1))
            s = coil_imgs[:, ix, iy].reshape((n_coils, 1))
            # Least squares: m = (S^H S)^-1 S^H s
            SHS = S.conj().T @ S  # scalar in 1x1
            SHs = S.conj().T @ s
            if np.abs(SHS) > 1e-6:
                m = SHs / SHS
                recon[ix, iy] = m
            else:
                recon[ix, iy] = 0
    return np.abs(recon)


def normalize_img(img):
    img = np.abs(img)
    img -= img.min()
    img /= img.max() + 1e-8
    return img

def create_cartesian_mask(shape, num_lines):
    mask = np.zeros(shape, dtype=bool)
    center = shape[1] // 2
    half_lines = num_lines // 2
    start = center - half_lines
    end = center + half_lines + (num_lines % 2)
    mask[:, start:end] = True
    return mask

img_size = 256
# -----------------------------
# Parameters
# -----------------------------
st.sidebar.title("MRI Parameters")
n_coils = st.sidebar.selectbox("Number of Coils", [1, 4, 8, 12, 16, 20, 24], index=1)
# n_coils = st.sidebar.slider("Number of Coils", 1, 12, 4)
coil_smoothness = st.sidebar.slider("Coil Sensitivity Smoothness",
                                0.1, 1.0, 0.5, step=0.05,
                                help="Controls spatial smoothness of coil sensitivity maps.")
acc_factor = st.sidebar.selectbox("Undersampling Factor (R)", [1, 2, 4], index=0)
# max_lines = img_size
# num_lines = st.sidebar.slider("Number of k-space Phase Encode Lines", 1, max_lines, max_lines,
#                           help="Number of phase-encode lines included in k-space reconstruction. Move slider to animate undersampling.")

# -----------------------------
# Ground Truth Phantom
# -----------------------------
phantom = resize(shepp_logan_phantom(), (img_size, img_size))
phantom = normalize_img(phantom)

# Simulated coil sensitivity maps
sens_maps = simulate_coil_sensitivities((img_size, img_size), n_coils, coil_smoothness)

# Coil images & k-space
coil_images = phantom * sens_maps
kspace_coils = fft2c(coil_images)

# Undersampling mask
mask = np.zeros_like(phantom)
mask[:, ::acc_factor] = 1
mask = np.tile(mask, (n_coils, 1, 1))

# Apply undersampling
kspace_undersampled = kspace_coils * mask

# Create mask and apply
# mask = create_cartesian_mask(kspace_undersampled.shape[-2:], num_lines)
# mask_3d = np.broadcast_to(mask, kspace_undersampled.shape)
# kspace_coils_masked = kspace_undersampled * mask_3d

kspace_coils_masked = kspace_undersampled
# Combined k-space magnitude
# combined_kspace_mag = np.sqrt(np.sum(np.abs(kspace_coils)**2, axis=0))

# SOS Reconstruction
sos_img = sos_reconstruction(ifft2c(kspace_coils_masked))

# SENSE Reconstruction
sense_img = sense_reconstruction(kspace_coils_masked, sens_maps)

# Metrics
psnr_sos = psnr(phantom, normalize_img(sos_img))
ssim_sos = ssim(phantom, normalize_img(sos_img), data_range=phantom.max()-phantom.min())
psnr_sense = psnr(phantom, normalize_img(sense_img))
ssim_sense = ssim(phantom, normalize_img(sense_img), data_range=phantom.max()-phantom.min())

# -----------------------------
# Layout with Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["K-space", "Coil Images", "Combination", "Final Results"])

with tab1:
    st.header("Ground Truth & Coil Sensitivity Effects")

    st.subheader("Ground Truth Image")
    st.image(normalize_img(phantom), caption="Shepp–Logan Phantom (Ground Truth)", clamp=True)

    st.subheader("Coil Sensitivity Maps")
    st.info("Each MRI receive coil has a different sensitivity pattern. "
            "These maps show how each coil's sensitivity varies across space.")
    images_per_row = 4
    for row_start in range(0, n_coils, images_per_row):
        cols = st.columns(min(images_per_row, n_coils - row_start))
        for idx, col in enumerate(cols):
            coil_idx = row_start + idx
            sens_map = normalize_img(sens_maps[coil_idx])
            with col:
                st.image(
                    sens_map,
                    caption=f"Coil {coil_idx+1} Sensitivity Map",
                    clamp=True,
                    use_container_width=True
                )

    st.subheader("K-space for Each Coil")
    st.info("The phantom is multiplied by each coil's sensitivity map to create a coil image. "
            "Taking the 2D FFT of each coil image gives that coil's k-space data.")
    # Grid display: adjustable number of images per row
    images_per_row = 4
    for row_start in range(0, n_coils, images_per_row):
        cols = st.columns(min(images_per_row, n_coils - row_start))
        for idx, col in enumerate(cols):
            coil_idx = row_start + idx
            kspace_mag = normalize_img(np.log1p(np.abs(kspace_coils_masked[coil_idx])))
            with col:
                st.image(
                    kspace_mag,
                    caption=f"Coil {coil_idx+1} K-space Magnitude",
                    clamp=True,
                    use_container_width=True
                )
with tab2:
    st.header("Coil-by-Coil Image Formation")
    st.subheader("K-space for Each Coil")
    st.info("The phantom is multiplied by each coil's sensitivity map to create a coil image. "
            "Taking the 2D FFT of each coil image gives that coil's k-space data.")
    # Grid display: adjustable number of images per row
    images_per_row = 4
    for row_start in range(0, n_coils, images_per_row):
        cols = st.columns(min(images_per_row, n_coils - row_start))
        for idx, col in enumerate(cols):
            coil_idx = row_start + idx
            kspace_mag = normalize_img(np.log1p(np.abs(kspace_coils_masked[coil_idx])))
            with col:
                st.image(
                    kspace_mag,
                    caption=f"Coil {coil_idx+1} K-space Magnitude",
                    clamp=True,
                    use_container_width=True
                )
    st.subheader("Inverse FFT to Coil Images")
    st.info("After inverse FFT of k-space, each coil produces its own image, still influenced by its sensitivity map.")
    # cols = st.columns(n_coils)
    # for i in range(n_coils):
    #     with cols[i]:
    #         st.image(normalize_img(ifft2c(kspace_coils_masked[i])),
    #                  caption=f"Coil {i+1} Image", clamp=True)
    
    images_per_row = 4
    for row_start in range(0, n_coils, images_per_row):
        cols = st.columns(min(images_per_row, n_coils - row_start))
        for idx, col in enumerate(cols):
            coil_idx = row_start + idx
            coil_image = normalize_img(ifft2c(kspace_coils_masked[coil_idx]))
            with col:
                st.image(
                    coil_image,
                    caption=f"Coil {coil_idx+1} Image",
                    clamp=True,
                    use_container_width=True
                )
    

with tab3:
    st.header("Combination Stage")
    st.info("Coil combination merges individual coil images into a final image. "
            "Sum-of-squares (SOS) is the simplest method, while SENSE uses coil sensitivity maps to unalias undersampled data.")
    st.subheader("SOS Intermediate")
    st.image(normalize_img(sos_img), caption="SOS Reconstruction", clamp=True, use_container_width=True)
    st.subheader("Coil Sensitivity Maps")
    cols = st.columns(n_coils)
    for i in range(n_coils):
        with cols[i]:
            st.image(normalize_img(sens_maps[i]),
                     caption=f"Coil {i+1} Sensitivity", clamp=True, use_container_width=True)

with tab4:
    st.header("Final Results & Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SOS Reconstruction")
        # Display SOS image with PSNR and SSIM metrics
        st.image(normalize_img(sos_img),
                 caption=f"SOS Reconstruction\nPSNR: {psnr_sos:.2f}, SSIM: {ssim_sos:.3f}", clamp=True, use_container_width=True)
        st.image(normalize_img(np.abs(phantom - normalize_img(sos_img))),
                 caption="Difference (SOS vs Ground Truth)", clamp=True, use_container_width=True)
        
        with st.expander("What is Sum-of-Squares (SoS) reconstruction?"):
            st.markdown("""
            **Sum-of-Squares (SoS) Reconstruction**

            - The simplest way to combine images from multiple receiver coils.
            - After reconstructing the individual coil images, you take the magnitude of each coil’s image, square them, sum them pixel-wise, and then take the square root.
            """)
            st.latex(r"I_{SoS}(x,y) = \sqrt{\sum_{c=1}^{N_c} \left|I_c(x,y)\right|^2}")
            st.markdown("""
            - Pros: Easy to implement, good SNR, no coil sensitivity maps required.
            - Cons: Does not exploit coil sensitivity for acceleration; cannot reduce scan time by undersampling.
            """)

    with col2:
        st.subheader("SENSE Reconstruction")
        st.image(normalize_img(sense_img),
                 caption=f"SENSE Reconstruction\nPSNR: {psnr_sense:.2f}, SSIM: {ssim_sense:.3f}", clamp=True, use_container_width=True)
        st.image(normalize_img(np.abs(phantom - normalize_img(sense_img))),
                 caption="Difference (SENSE vs Ground Truth)", clamp=True, use_container_width=True)
        with st.expander("What is SENSE reconstruction?"):
            st.markdown("""
            **SENSE (Sensitivity Encoding) Reconstruction**

            - A parallel imaging technique that uses coil sensitivity maps to reconstruct images from undersampled k-space.
            - “Unfolds” aliasing caused by undersampling by solving a pixel-wise linear system.
            """)
            # SENSE forward model
            st.latex(r"\mathbf{s} = \mathbf{S} \cdot m(x,y) + \text{noise}")

            # SENSE reconstruction formula
            st.latex(r"m(x,y) = \left( \mathbf{S}^H \mathbf{S} \right)^{-1} \mathbf{S}^H \mathbf{s}")       
            st.markdown("""
            - Pros: Allows accelerated MRI scans, improves imaging speed, reduces aliasing.
            - Cons: Requires accurate coil sensitivity maps, more computationally complex, sensitive to errors.
            """)

