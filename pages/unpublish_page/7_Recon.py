import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import zoom
import io
import base64
import time

# ------------------------------
# Helper utilities
# ------------------------------
def array_to_png_base64(arr, cmap='gray', vmin=None, vmax=None):
    """Render a numpy 2D array to PNG bytes and return a base64 data URI (for HTML <img title="...">)."""
    fig = plt.figure(figsize=(3,3), dpi=85)
    ax = fig.add_subplot(111)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax.axis('off')
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{data}"

def add_noise(sinogram, noise_level):
    """Add Gaussian noise to the sinogram."""
    if noise_level > 0:
        sigma = noise_level * np.max(sinogram)
        noise = np.random.normal(0, sigma, sinogram.shape)
        return sinogram + noise
    return sinogram

# def filtered_backprojection(sinogram, theta, filter_type, cutoff, circle=True):
#     """FBP reconstruction using scikit-image's iradon (wrap)."""
#     return iradon(sinogram, theta=theta, filter_name=filter_type, frequency_scaling=cutoff, circle=circle)

def filtered_backprojection(sinogram, theta, filter_type, circle=True):
    """FBP reconstruction for your scikit-image version."""
    return iradon(
        sinogram,
        theta=theta,
        filter_name=filter_type,
        circle=circle
    )

def mlem(sinogram, theta, iterations, circle=True):
    """MLEM reconstruction (educational version). Returns (reco, rmse_list)."""
    sinogram = np.maximum(sinogram, 1e-12)
    img_size = sinogram.shape[0]
    reco = np.ones((img_size, img_size))
    # backproj_ones = iradon(np.ones_like(sinogram), theta=theta, filter_name=None, circle=circle)
    backproj_ones = iradon(np.ones_like(sinogram), theta=theta, filter_name=None, circle=circle)
    rmse_list = []
    for _ in range(iterations):
        forward_proj = radon(reco, theta=theta, circle=circle)
        ratio = sinogram / np.maximum(forward_proj, 1e-12)
        correction = iradon(ratio, theta=theta, filter_name=None, circle=circle)
        reco *= correction / np.maximum(backproj_ones, 1e-12)
        rmse_list.append(np.sqrt(np.mean((reco - phantom) ** 2)))
    return reco, rmse_list

def compute_metrics(gt, recon):
    rmse_value = np.sqrt(np.mean((recon - gt) ** 2))
    ssim_value = ssim(gt, recon, data_range=recon.max() - recon.min())
    return rmse_value, ssim_value

def highlight_sinogram_column(sinogram, col_idx):
    """Return image where selected column is highlighted (red overlay) for visualization."""
    # Normalize sinogram for plotting base
    base = sinogram.copy()
    base_norm = (base - base.min()) / (base.max() - base.min() + 1e-12)
    # create an RGB image (H,W,3)
    rgb = np.stack([base_norm, base_norm, base_norm], axis=-1)
    # highlight column with a red vertical stripe
    if 0 <= col_idx < rgb.shape[1]:
        rgb[:, col_idx, 0] = 1.0  # red channel
        rgb[:, col_idx, 1] = 0.0
        rgb[:, col_idx, 2] = 0.0
    return rgb

# ------------------------------
# App setup
# ------------------------------
st.set_page_config(page_title="CT Reconstruction Teaching App", layout="wide")
st.title("🩻 CT Reconstruction — Teaching Prototype (256×256)")
st.markdown("Interactive demo: FBP, MLEM, sinogram viewer, stepwise backprojection, difference map & metrics.")

# Sidebar: acquisition and algorithm
st.sidebar.header("Acquisition & Algorithm")
num_angles = st.sidebar.slider("Number of projection angles", 10, 180, 90,
                               help="More angles → better quality, fewer artifacts.")
noise_level = st.sidebar.slider("Noise level (fraction of max sinogram)", 0.0, 0.12, 0.02, 0.01,
                                help="Gaussian noise level to simulate low-dose acquisition.")
algorithm = st.sidebar.selectbox("Algorithm", ["Filtered Backprojection (FBP)", "Simple Backprojection", "MLEM"])

# Algorithm-specific params
if algorithm == "Filtered Backprojection (FBP)":
    st.sidebar.subheader("FBP parameters")
    filter_type = st.sidebar.selectbox("Filter type", ["ramp", "shepp-logan", "cosine", "hamming", "hann"],
                                       help="Frequency-domain filter used before backprojection.")
    # cutoff = st.sidebar.slider("Cutoff frequency (fraction of Nyquist)", 0.1, 1.0, 1.0, 0.05,
    #                            help="Lower cutoff reduces noise but blurs detail.")
elif algorithm == "Simple Backprojection":
    st.sidebar.subheader("Simple Backprojection parameters")
    filter_type = None  # No filtering in simple backprojection
    cutoff = None  # Not used
    st.sidebar.info("Simple backprojection does not use filtering; it directly backprojects the sinogram.")
else:
    st.sidebar.subheader("MLEM parameters")
    iterations = st.sidebar.slider("Iterations", 1, 60, 12,
                                   help="More iterations usually increase detail but also noise.")
    # For stepwise backprojection mode we also use 'subangles' slider below.

# ------------------------------
# Phantom & Sinogram (256x256)
# ------------------------------
phantom = zoom(shepp_logan_phantom(), 256 / shepp_logan_phantom().shape[0])
theta = np.linspace(0., 180., num_angles, endpoint=False)
sinogram = radon(phantom, theta=theta, circle=True)
sinogram_noisy = add_noise(sinogram, noise_level)

# ------------------------------
# Top: main controls for teaching
# ------------------------------
st.subheader("Interactive viewers")
top_cols = st.columns([2, 1])
with top_cols[0]:
    st.markdown("**Sinogram viewer & projection inspector**")
    # slider to pick angle index
    angle_idx = st.slider("Projection (angle) index", 0, max(0, num_angles - 1), int(num_angles//2),
                          help="Select a projection angle to inspect. The selected column is highlighted on the sinogram.")
    sin_highlight = highlight_sinogram_column(sinogram_noisy, angle_idx)
    # render sinogram as embedded <img> with title tooltip
    sin_img_uri = array_to_png_base64((sin_highlight * 255).astype(np.uint8), cmap='gray')  # rgb already
    sin_tooltip = f"Sinogram (noisy). Column {angle_idx} = projection at angle {theta[angle_idx]:.1f}°"
    st.markdown(f'<img src="{sin_img_uri}" title="{sin_tooltip}" style="width:100%;height:auto;border:1px solid #ddd;border-radius:4px;">',
                unsafe_allow_html=True)

    # plot the selected projection (1D)
    proj = sinogram_noisy[:, angle_idx]
    fig1, ax1 = plt.subplots(figsize=(6,1.8), dpi=90)
    ax1.plot(proj)
    ax1.set_title(f"Projection at angle {theta[angle_idx]:.1f}° (index {angle_idx})")
    ax1.set_xlabel("Detector position")
    ax1.set_ylabel("Counts")
    plt.tight_layout()
    st.pyplot(fig1)

with top_cols[1]:
    st.markdown("**Step-by-step Backprojection (teaching)**")
    st.markdown("Use the slider to include only the first *N* projection angles for backprojection. This shows how artefacts fill in as angles accumulate.")
    n_steps = st.slider("Number of projections used for progressive backprojection", 1, max(1, num_angles), min(10, max(1, num_angles)),
                        help="Use the first N projections (sorted by angle) to form a progressive backprojection.")
    animate = st.button("Animate progressive backprojection")

# ------------------------------
# Progressive/backprojection function
# ------------------------------
# def progressive_backprojection(sino, theta, n_projs, filter_type=None, cutoff=1.0):
#     """
#     Backproject using only the first n_projs projections.
#     If filter_type provided, apply that filtering (FBP-like behavior), else use simple unfiltered backprojection.
#     """
#     # Create a sinogram with zeros for the unused angles
#     mask = np.zeros_like(sino)
#     mask[:, :n_projs] = sino[:, :n_projs]
#     # If filter_type is provided, use iradon with filter (FBP with truncated angles).
#     if filter_type is not None:
#         # Use the subset of theta matching columns kept
#         theta_subset = theta[:n_projs]
#         # iradon expects sinogram with same number of angles as theta; provide only the subset
#         recon = iradon(mask[:, :n_projs], theta=theta_subset, filter_name=filter_type, frequency_scaling=cutoff, circle=True)
#     else:
#         # use unfiltered backprojection: call iradon with filter_name=None for the subset
#         theta_subset = theta[:n_projs]
#         recon = iradon(mask[:, :n_projs], theta=theta_subset, filter_name=None, circle=True)
#     return recon

def progressive_backprojection(sino, theta, n_projs, filter_type=None):
    """
    Step-by-step backprojection for teaching.
    n_projs = number of projections to use from the sinogram.
    """
    theta_subset = theta[:n_projs]
    sino_subset = sino[:, :n_projs]
    return iradon(
        sino_subset,
        theta=theta_subset,
        filter_name=filter_type,
        circle=True
    )

# Animate if requested (updates in-place)
if animate:
    placeholder = st.empty()
    # small animation loop
    for k in range(1, n_steps + 1):
        recon_step = progressive_backprojection(sinogram_noisy, theta, k,
                                                filter_type if algorithm.startswith("Filtered") else None)
        png = array_to_png_base64(recon_step, cmap='gray')
        tooltip = f"Progressive backprojection using first {k}/{num_angles} angles."
        with placeholder.container():
            st.markdown(f'<img src="{png}" title="{tooltip}" style="width:100%;height:auto;border:1px solid #ddd;border-radius:4px;">', unsafe_allow_html=True)
            st.write(f"Using {k}/{num_angles} projections")
        time.sleep(0.12)

# Show final progressive step (non-animated) as user-set n_steps
recon_progress = progressive_backprojection(sinogram_noisy, theta, n_steps,
                                            filter_type if algorithm.startswith("Filtered") else None)
st.markdown("**Progressive backprojection (current selection)**")
prog_png = array_to_png_base64(recon_progress, cmap='gray')
st.markdown(f'<img src="{prog_png}" title="Progressive backprojection using first {n_steps} projections." style="width:100%;height:auto;border:1px solid #ddd;border-radius:4px;">',
            unsafe_allow_html=True)

# ------------------------------
# Perform full reconstruction according to chosen algorithm
# ------------------------------
st.subheader("Full reconstruction (using all projections / chosen algorithm)")
if algorithm == "Filtered Backprojection (FBP)":
    reconstruction = filtered_backprojection(sinogram_noisy, theta, filter_type)
    recon_tooltip = f"Filtered Backprojection (filter={filter_type}) using {num_angles} angles."
elif algorithm == "MLEM":
    reconstruction, rmse_list = mlem(sinogram_noisy, theta, iterations)
    recon_tooltip = f"MLEM reconstruction ({iterations} iterations) using {num_angles} angles."

else:  # Simple Backprojection
    reconstruction = filtered_backprojection(sinogram_noisy, theta, filter_type=None)
    recon_tooltip = f"Simple Backprojection using {num_angles} angles (no filtering)."
    
# Embed images with HTML tooltips
phantom_uri = array_to_png_base64(phantom, cmap='gray')
phantom_tooltip = "Original phantom (ground truth)."
sinogram_uri = array_to_png_base64((sinogram_noisy / np.max(sinogram_noisy) * 255).astype(np.uint8), cmap='gray')
sinogram_tooltip = "Sinogram (noisy). The vertical axis is detector position, horizontal axis is angle index."
recon_uri = array_to_png_base64(reconstruction, cmap='gray')
recon_tooltip = recon_tooltip
difference = reconstruction - phantom
diff_abs = np.abs(difference)

# Layout: three panels (phantom, sinogram, reconstruction) with inline tooltips
colA, colB, colC = st.columns(3)
with colA:
    st.markdown("**Ground truth**")
    st.markdown(f'<img src="{phantom_uri}" title="{phantom_tooltip}" style="width:100%;height:auto;border:1px solid #ddd;border-radius:4px;">', unsafe_allow_html=True)
with colB:
    st.markdown("**Sinogram (noisy)**")
    st.markdown(f'<img src="{sinogram_uri}" title="{sinogram_tooltip}" style="width:100%;height:auto;border:1px solid #ddd;border-radius:4px;">', unsafe_allow_html=True)
with colC:
    st.markdown("**Reconstruction**")
    st.markdown(f'<img src="{recon_uri}" title="{recon_tooltip}" style="width:100%;height:auto;border:1px solid #ddd;border-radius:4px;">', unsafe_allow_html=True)

# ------------------------------
# Difference image + metrics panel
# ------------------------------
st.subheader("Difference map & metrics")
diff_uri = array_to_png_base64(diff_abs, cmap='gray', vmin=0, vmax=np.max(diff_abs) if np.max(diff_abs)>0 else 1)
st.markdown("Absolute difference (|recon − ground truth|)")
st.markdown(f'<img src="{diff_uri}" title="Absolute difference between reconstruction and ground truth." style="width:40%;height:auto;border:1px solid #ddd;border-radius:4px;">', unsafe_allow_html=True)

rmse_value, ssim_value = compute_metrics(phantom, reconstruction)
st.markdown(f"**RMSE:** `{rmse_value:.5f}` &nbsp;&nbsp; **SSIM:** `{ssim_value:.4f}`")

# Error histogram
fig2, ax2 = plt.subplots(figsize=(6,2.5), dpi=90)
ax2.hist((reconstruction - phantom).ravel(), bins=60)
ax2.set_title("Error histogram (reconstruction - ground truth)")
ax2.set_xlabel("Error")
ax2.set_ylabel("Count")
plt.tight_layout()
st.pyplot(fig2)

# If MLEM, show convergence plot
if algorithm == "MLEM":
    fig3, ax3 = plt.subplots(figsize=(5,2.2), dpi=90)
    ax3.plot(range(1, len(rmse_list) + 1), rmse_list, marker='o')
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("RMSE")
    ax3.set_title("MLEM Convergence (RMSE vs iteration)")
    st.pyplot(fig3)

# ------------------------------
# Explanations / quicktips (collapsible)
# ------------------------------
st.markdown("---")
with st.expander("Parameter tooltips & short explanations (click to expand)"):
    st.markdown("""
    **Number of projection angles** — how many angular views were collected (0..180°).  
    **Noise level** — Gaussian noise added to the sinogram; simulates low-dose acquisition.  
    **FBP Filter type** — frequency-domain filter: `ramp` (preserves high-frequency detail), `shepp-logan` (smooths high-freq), `hann`/`hamming` (windowed).  
    **Cutoff frequency** — fraction of Nyquist to keep: lower values reduce noise but blur detail.  
    **MLEM Iterations** — more iterations usually refine structure but amplify noise; check the convergence plot.
    """)
    st.markdown("**Image tooltips:** Hover the images to see brief explanations (browser-native tooltip).")

st.caption("Prototype: educational/demo use. For real clinical recon use optimized libraries (ASTRA, BART, STIR) and validated workflows.")