import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import radon
from scipy import ndimage
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly import colors
from PIL import Image

st.set_page_config(page_title="Sinogram", page_icon="✋🏻", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .stAlert {
        margin-top: -1rem;
    }
    h3 {
        margin-top: 2rem;
        color: #007BFF;
    }
    </style>
    """, unsafe_allow_html=True)

# ---- HEADER SECTION ----
st.title("Profile, Projection, and Sinogram")
st.markdown("---")

# ---- LOAD DATA ----
BASE_DIR = Path(__file__).resolve().parent
PRJ_DIR = BASE_DIR / 'images/sino_proj/Shepp_Logan_Prj.npy'
SLICE_DIR = BASE_DIR / 'images/sino_proj/Shepp_Logan_copy.npy'
SINO_DIR = BASE_DIR / 'images/sino_proj/shepp_logan_sinogram.npy'

prj, tomo, sinogram_raw = np.load(PRJ_DIR), np.load(SLICE_DIR), np.load(SINO_DIR)
n_ang, n_z, n_x = np.shape(prj)

# Apply Motion Simulation Logic
def apply_motion(sino, amplitude):
    if amplitude == 0: return sino
    shifted_sino = np.zeros_like(sino)
    for i in range(sino.shape[0]):
        shift = np.random.randint(-amplitude, amplitude + 1)
        shifted_sino[i, :] = np.roll(sino[i, :], shift)
    return shifted_sino

# ---- 1️⃣ STEP 1: PROJECTION ACQUISITION ----
st.subheader("1️⃣ Step 1: 2D Projection Acquisition")
st.info("The detector rotates around the patient, capturing 2D 'shadow' images (projections) at each angle.")

with st.form("Acquisition Settings"):
    acq_col1, acq_col2 = st.columns([1, 2])
    with acq_col1:
        start_ang = st.radio("Start angle (ϴ):", [0, 45, 90, 180], index=0, horizontal=True)
        step_ang = st.radio("Step angle (ϴ):", [3, 6, 12], index=2, horizontal=True)
        ang_range = st.radio("Angular range (ϴ):", [180, 360], index=1, horizontal=True)
        rot_dir = st.radio("Rotation Direction:", ["CW", "CCW"], index=0, horizontal=True)
        btn_acq = st.form_submit_button("Preview Acquisition")
    
    with acq_col2:
        # Acquisition Logic
        if rot_dir == 'CW':
            theta_acq = np.array(range(start_ang, start_ang + ang_range, step_ang))
            wrapped_range = [(start_ang + i) % 360 for i in range(0, ang_range, step_ang)]
        else:
            theta_acq = np.array(range(start_ang, start_ang - ang_range, -step_ang))
            wrapped_range = [(start_ang - i) % 360 for i in range(0, ang_range, step_ang)]
        
        # Extract corresponding 2D projections from the 3D volume
        ang_indices_3deg = range(357, -1, -3) # Data is 0-357 in 3 deg steps
        ang_list = list(ang_indices_3deg)
        proj_indices = [ang_list.index(v) for v in wrapped_range if v in ang_list]
        
        prj_subset = np.zeros((len(proj_indices), n_z, n_x))
        for i, idx in enumerate(proj_indices):
            prj_subset[i, :, :] = prj[idx, :, :]
        
        fig_prj_anim = px.imshow(prj_subset, animation_frame=0, binary_string=True, 
                               labels=dict(animation_frame="Projection Index", x="Detector X", y="Detector Y"))
        fig_prj_anim.update_layout(width=400, height=400, margin=dict(l=0,r=0,t=0,b=0))
        fig_prj_anim.update_xaxes(title_text="Detector Position (X)").update_yaxes(title_text="Detector Position (Y)")
        st.plotly_chart(fig_prj_anim, use_container_width=True)

# ---- 2️⃣ STEP 2: THE INTERACTIVE BRIDGE (PROFILE & SINOGRAM) ----
st.subheader("2️⃣ Step 2: Interactive Exploration")
st.info("Adjust the anatomical slice and simulate motion to see how it affects the raw data (profiles and sinograms).")

# --- Interactive Controls ---
with st.container():
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1], gap="medium")
    with ctrl_col1:
        explorer_angle = st.slider("Scrub Acquisition Angle (ϴ):", 0, 357, 0, step=3)
    with ctrl_col2:
        selected_slice = st.slider("Select Slice (Z):", 0, n_z - 1, n_z // 2)
    with ctrl_col3:
        simulate_motion = st.checkbox("Simulate Motion")
        motion_amplitude = st.slider("Amplitude:", 1, 10, 3) if simulate_motion else 0

# --- Generate Dynamic Sinogram from 3D Projections ---
# prj has shape (120, n_z, n_x), where 120 are angles in 3-degree steps
sinogram_dynamic = np.zeros((360, n_x))
ang_indices_3deg = range(357, -1, -3) 
ang_list = list(ang_indices_3deg)

for i, ang in enumerate(ang_list):
    sinogram_dynamic[ang, :] = prj[i, selected_slice, :]

# Apply Motion to the dynamic sinogram
sinogram = apply_motion(sinogram_dynamic, motion_amplitude)
# ----------------------------------------------------

bridge_col1, bridge_col2 = st.columns([1, 1], gap="large")

with bridge_col1:
    st.markdown("**A. Relative Motion Concept**")
    perspective = st.radio("Perspective:", ["Room View", "Detector View"], horizontal=True)
    
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        if perspective == "Room View":
            st.caption("Patient (Fixed) & Detector (Moving)")
            fig_room = px.imshow(tomo, binary_string=True, labels=dict(x="X Position", y="Y Position"))
            rad = np.deg2rad(90 - explorer_angle)
            cx, cy = n_x // 2, n_z // 2
            r_dist = n_x // 2 + 15
            dx = cx + r_dist * np.cos(rad)
            dy = cy - r_dist * np.sin(rad)
            half_len = n_x // 2
            x0, y0 = dx + half_len * np.sin(rad), dy + half_len * np.cos(rad)
            x1, y1 = dx - half_len * np.sin(rad), dy - half_len * np.cos(rad)
            fig_room.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="red", width=6))
            fig_room.add_trace(go.Scatter(x=[dx], y=[dy], mode='text', text=[f"{explorer_angle}°"], textfont=dict(color="red"), showlegend=False))
            fig_room.update_xaxes(title_text="X Position").update_yaxes(title_text="Y Position")
            st.plotly_chart(fig_room, use_container_width=True)
        else:
            st.caption("Detector (Fixed at Top)")
            rotated_slice = ndimage.rotate(tomo, explorer_angle, reshape=False)
            fig_slice = px.imshow(rotated_slice, binary_string=True, labels=dict(x="X'", y="Y'"))
            fig_slice.add_shape(type="rect", x0=0, y0=-10, x1=n_x, y1=-2, fillcolor="red", line=dict(color="red"), opacity=0.8)
            fig_slice.update_xaxes(title_text="X' (Rotating)").update_yaxes(title_text="Y' (Rotating)")
            st.plotly_chart(fig_slice, use_container_width=True)
    
    with sub_col2:
        st.caption(f"Detector Output ({explorer_angle}°)")
        p_idx = ang_list.index(explorer_angle)
        p_img = prj[p_idx, :, :].copy()
        if simulate_motion:
            p_img = np.roll(p_img, np.random.randint(-motion_amplitude, motion_amplitude+1), axis=1)
        fig_p = px.imshow(p_img, binary_string=True, labels=dict(x="Detector X", y="Detector Y"))
        fig_p.add_hline(y=selected_slice, line_dash="dash", line_color="red")
        fig_p.update_xaxes(title_text="Detector Position (X)").update_yaxes(title_text="Detector Position (Y)")
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("**B. 1D Profile Extraction**")
    profile = p_img[selected_slice, :]
    st.caption("Profile Image (1D)")
    fig_prof_img = px.imshow(profile[np.newaxis, ...], binary_string=True, labels=dict(x="Detector X", y="Intensity"))
    fig_prof_img.update_layout(height=120, margin=dict(l=0,r=0,t=0,b=40))
    fig_prof_img.update_xaxes(title_text="Detector Position").update_yaxes(showticklabels=False)
    st.plotly_chart(fig_prof_img, use_container_width=True)

    st.caption("Profile Plot (Line)")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(y=profile, line=dict(color='#007BFF', width=3)))
    fig_line.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=0), xaxis_title="Detector Position", yaxis_title="Pixel Intensity")
    st.plotly_chart(fig_line, use_container_width=True)

with bridge_col2:
    st.markdown("**C. Sinogram Construction**")
    fig_sino_bridge = px.imshow(sinogram, color_continuous_scale='gray', labels=dict(x="Detector X", y="Angle (θ)"))
    fig_sino_bridge.add_hline(y=explorer_angle, line_dash="solid", line_color="red", annotation_text=f"{explorer_angle}°", annotation_position="top right")
    fig_sino_bridge.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=500, coloraxis_showscale=False)
    fig_sino_bridge.update_xaxes(title_text="Detector Position (X)").update_yaxes(title_text="Acquisition Angle (θ)")
    st.plotly_chart(fig_sino_bridge, use_container_width=True)
    st.caption("The red line highlights the current angle in the complete Sinogram.")

# ---- FOOTER ----
st.markdown("---")
st.caption("Anucha Chaichana | anucha.cha@mahidol.ac.th")
