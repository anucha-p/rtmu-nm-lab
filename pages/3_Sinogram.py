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
        
        # Data Info Display
        total_projections = len(proj_indices)
        st.write(f"**Total Projections Acquired:** {total_projections}")
        st.write(f"**Detector Matrix Size:** {n_x} (bins) x {n_z} (slices)")
        
        # Grid/Montage of Projections
        st.markdown("**Capturing Sequence (Montage)**")
        
        # Show exactly 8 projections (or total if less than 8)
        num_m = min(8, total_projections)
        m_step = max(1, total_projections // num_m)
        montage_indices = proj_indices[::m_step][:num_m]
        
        n_rows_m = 2
        n_cols_m = 4
        
        fig_m1 = make_subplots(rows=n_rows_m, cols=n_cols_m, 
                              subplot_titles=[f"{ang_list[idx]}°" for idx in montage_indices],
                              horizontal_spacing=0.01, vertical_spacing=0.1)
        
        for i, idx in enumerate(montage_indices):
            r, c = (i // n_cols_m) + 1, (i % n_cols_m) + 1
            fig_m1.add_trace(go.Heatmap(z=prj[idx, :, :], colorscale='gray', showscale=False), row=r, col=c)
        
        fig_m1.update_xaxes(showticklabels=False)
        fig_m1.update_yaxes(
            showticklabels=False, 
            autorange='reversed',
            scaleanchor="x",
            scaleratio=1
        )
        fig_m1.update_layout(height=450, width=900, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_m1, use_container_width=False)

        st.markdown("**Animation of All Captured Projections**")
        # Custom frame labels for animation
        frame_labels = [f"#{i+1} ({ang}°)" for i, ang in enumerate(wrapped_range)]
        
        fig_prj_anim = px.imshow(prj_subset, animation_frame=0, binary_string=True, 
                               labels=dict(animation_frame="Projection Angle", x="Detector Position (X)", y="Detector Position (Y)"))
        
        # Manually update slider labels to show degrees
        for i, label in enumerate(frame_labels):
            fig_prj_anim.layout.sliders[0].steps[i].label = label

        fig_prj_anim.update_layout(width=400, height=400, margin=dict(l=0,r=0,t=0,b=0))
        fig_prj_anim.update_xaxes(title_text="Detector X").update_yaxes(title_text="Detector Y")
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

# ---- 3️⃣ STEP 3: ADVANCED MONTAGE & SINOGRAM EXPLORER ----
st.subheader("3️⃣ Step 3: Projection & Sinogram Linkage")
st.info("""
In this view, we look at **multiple projections** simultaneously. 
- **Green Lines:** Show the same anatomical slice across all angles.
- **Red Line:** Highlights the specific angle you are currently focusing on in the montage grid.
""")

# Interaction for Step 3
m_c1, m_c2 = st.columns([1, 1])
with m_c1:
    y_sel = st.slider("Select Slice Row (Y):", 0, n_z - 1, selected_slice, key="y_sel")
with m_c2:
    # Use a subset of angles for the montage to keep it readable (every 30 degrees)
    montage_angles = sorted([a for a in ang_list if a % 30 == 0])
    a_sel = st.select_slider("Select Active Angle (ϴ) in Montage:", options=montage_angles, value=0)

exp_col1, exp_col2 = st.columns([1.2, 0.8], gap="large")

with exp_col1:
    st.markdown("**Projection Montage**")
    # Create 3x4 grid for 12 projections
    n_rows, n_cols = 3, 4
    fig_montage = make_subplots(rows=n_rows, cols=n_cols, 
                               subplot_titles=[f"{ang}°" for ang in montage_angles],
                               horizontal_spacing=0.02, vertical_spacing=0.05)
    
    for i, ang in enumerate(montage_angles):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        # Get projection data
        p_idx = ang_list.index(ang)
        p_data = prj[p_idx, :, :]
        
        # Add as heatmap
        fig_montage.add_trace(go.Heatmap(z=p_data, colorscale='gray', showscale=False), row=row, col=col)
        
        # Add highlight lines
        line_color = "red" if ang == a_sel else "green"
        fig_montage.add_shape(type="line", x0=0, y0=y_sel, x1=n_x, y1=y_sel, 
                             line=dict(color=line_color, width=2, dash="dash"),
                             row=row, col=col)
        
    fig_montage.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False, autorange='reversed')
    fig_montage.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_montage, use_container_width=True)

with exp_col2:
    st.markdown("**Dynamic Sinogram**")
    # Rebuild sinogram for the y_sel
    sino_exp = np.zeros((360, n_x))
    for i, ang in enumerate(ang_list):
        sino_exp[ang, :] = prj[i, y_sel, :]
    
    if simulate_motion:
        sino_exp = apply_motion(sino_exp, motion_amplitude)
        
    fig_sino_exp = px.imshow(sino_exp, color_continuous_scale='gray',
                            labels=dict(x="Detector Position (X)", y="Angle (θ)"))
    
    # Highlight row
    fig_sino_exp.add_hline(y=a_sel, line_dash="solid", line_color="red",
                          annotation_text=f"Focus: {a_sel}°", annotation_position="top right")
    
    fig_sino_exp.update_layout(height=600, margin=dict(l=0,r=0,t=0,b=0), coloraxis_showscale=False)
    fig_sino_exp.update_yaxes(autorange='reversed', title_text="Acquisition Angle (θ)")
    fig_sino_exp.update_xaxes(title_text="Detector Position (X)")
    st.plotly_chart(fig_sino_exp, use_container_width=True)
    st.caption(f"The red line highlights the specific angle ({a_sel}°) you are focusing on in the montage.")

# ---- FOOTER ----
st.markdown("---")
st.caption("Anucha Chaichana | anucha.cha@mahidol.ac.th")
