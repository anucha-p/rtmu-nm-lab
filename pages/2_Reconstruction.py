from skimage.transform import radon, iradon
from skimage.filters import gaussian
# from skimage.draw import disk
import streamlit as st
# from pathlib import Path

import numpy as np
# import pydicom as dicom
# import math
import pandas as pd
import altair as alt
# import streamlit_nested_layout
import time
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

from src.Filters import *

st.set_page_config(page_title="Reconstruction", page_icon="✋🏻", layout="wide")

remove_top_padding = """
        <style>
            .css-18e3th9 {
                padding-bottom: 10rem;
                padding-top: 0rem;
                padding-left: 2.5rem;
                padding-right: 2.5rem;
                }
            .css-1d391kg {
                padding-top: 3.5rem;
                padding-right: 1rem;
                padding-bottom: 3.5rem;
                padding-left: 1rem;
                }
        </style>
        """
st.markdown(remove_top_padding, unsafe_allow_html=True)

# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)

# ---- HEADER SECTION ----
with st.container():
    st.title("Image Reconstruction in Nuclear Medicine")
    st.write("---")
# ---- LOAD IMAGE ----
# BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = os.path.abspath(os.path.join(__file__, '../'))
IMAGE_DIR = os.path.join(BASE_DIR, 'images/recon')
# IMAGE_DIR = BASE_DIR / 'images/recon' 
# imageNames = [f.name for f in IMAGE_DIR.iterdir() if f.name.endswith('.npy')]
file_name = os.listdir(IMAGE_DIR)
imageNames = [f for f in file_name if f.endswith('.npy')]
st.sidebar.header("Reconstruction")


def mlem(sinogram, niter, progress_bar=None, status_text=None, image_placeholder=None):
    tt = time.time()
    image_shape, nview = np.shape(sinogram)
    theta = np.linspace(0.0, 360.0, nview, endpoint=False)
    mlem_rec = np.ones([image_shape, image_shape])
    sino_ones = np.ones(sinogram.shape)
    sens_image = iradon(sino_ones, theta=theta, circle=True, filter_name=None)
    
    convergence = []

    for i in range(niter):
        if status_text:
            status_text.text(f"MLEM Iteration {i+1}/{niter}...")
        
        prev_rec = mlem_rec.copy()
        fp = radon(mlem_rec, theta, circle=True)
        ratio = sinogram / (fp + 0.000001)
        correction = iradon(ratio, theta, circle=True,
                            filter_name=None) / (sens_image+0.00000001)
        mlem_rec = mlem_rec * correction
        
        # Calculate RMSE as a convergence metric
        rmse = np.sqrt(np.mean((mlem_rec - prev_rec)**2))
        convergence.append(rmse)
        
        if progress_bar:
            progress_bar.progress((i + 1) / niter)
        if image_placeholder:
            disp = (mlem_rec - mlem_rec.min()) / (mlem_rec.max() - mlem_rec.min() + 1e-9)
            image_placeholder.image(disp, caption=f"MLEM Iteration {i+1}", use_container_width=False, width=340)

    elapsed = (time.time() - tt)
    elapsed = "{:.2f}".format(elapsed)
    st.info('Reconstruction Completed; Total time (sec): ' + str(elapsed))

    return mlem_rec, convergence


def osem(sinogram, niter, nsub, progress_bar=None, status_text=None, image_placeholder=None):
    tt = time.time()
    image_shape, nview = np.shape(sinogram)
    theta = np.linspace(0.0, 360.0, nview, endpoint=False)
    
    if nview % nsub != 0:
        return None, []

    osem_rec = np.ones([image_shape, image_shape])
    sino_ones = np.ones(sinogram.shape)
    sens_image = iradon(sino_ones, theta=theta, circle=True, filter_name=None)
    
    convergence = []
    
    for i in range(niter):
        if status_text:
            status_text.text(f"OSEM Iteration {i+1}/{niter}...")
            
        prev_rec = osem_rec.copy()
        order_sub = np.random.permutation(range(nsub))
        for sub in order_sub:
            views = range(sub, nview, nsub)
            fp = radon(osem_rec, theta[views], circle=True)
            ratio = sinogram[:, views] / (fp + 0.000001)
            correction = iradon(
                ratio, theta[views], circle=True, filter_name=None) / (sens_image+0.00000001)
            osem_rec = osem_rec * correction
        
        # RMSE
        rmse = np.sqrt(np.mean((osem_rec - prev_rec)**2))
        convergence.append(rmse)
        
        if progress_bar:
            progress_bar.progress((i + 1) / niter)
        if image_placeholder:
            disp = (osem_rec - osem_rec.min()) / (osem_rec.max() - osem_rec.min() + 1e-9)
            image_placeholder.image(disp, caption=f"OSEM Iteration {i+1} (Subsets: {nsub})", use_container_width=False, width=340)

    elapsed = (time.time() - tt)
    elapsed = "{:.2f}".format(elapsed)
    st.info('Reconstruction Completed; Total time (sec): ' + str(elapsed))
    return osem_rec, convergence


# @st.cache_data(ttl=60, max_entries=10, show_spinner="Reconstruction in progress...")
# @st.cache_data(max_entries=1)
def fbp(measured_sino, filter_name='ramp', arc=360):
    tt = time.time()
    x, t = np.shape(measured_sino)
    proj_angles = np.linspace(0, arc, t, endpoint=False)
    backproj = iradon(measured_sino, proj_angles, filter_name=filter_name)
    elapsed = (time.time() - tt)
    elapsed = "{:.2f}".format(elapsed)
    st.info('Reconstruction Completed; Total time (sec): ' + str(elapsed))
    return backproj


# @st.cache_data(ttl=60, max_entries=10, show_spinner="Reconstruction in progress...")
# @st.cache_data(max_entries=1)
def bp(measured_sino, arc=360):
    tt = time.time()
    x, t = np.shape(measured_sino)
    proj_angles = np.array(range(0, arc, int(arc/t)))
    backproj = iradon(measured_sino, proj_angles, filter_name=None)
    elapsed = (time.time() - tt)
    elapsed = "{:.2f}".format(elapsed)
    st.info('Reconstruction Completed; Estimated time (sec): ' + str(elapsed))
    return backproj


# @st.cache_data(max_entries=2)
# def read_dcm(dcm_img):
#     ds = dicom.dcmread(dcm_img)
#     img = ds.pixel_array.astype(float)
#     return img

# @st.cache_data(max_entries=2)
def read_sino(sino_npy_file):
    sino = np.load(sino_npy_file)
    return sino

# @st.cache_data(max_entries=2)
def get_disp_img(img):
    scaled_image = (np.maximum(img, 0) / img.max()) * 255.0
    disp_img = np.uint8(scaled_image)
    return disp_img

# ------  APP -----#
# # if img_path is not None:
# img = read_dcm(img_path)
# disp_img = get_disp_img(img)
# t,m,n = np.shape(img)
# x = np.linspace(0,1,int(m/2))

# ---- STORE IMAGES IN STATE
if 'compare_image' not in st.session_state:
    st.session_state.compare_image = []
if 'last_recon' not in st.session_state:
    st.session_state.last_recon = None
if 'last_recon_str' not in st.session_state:
    st.session_state.last_recon_str = ""

Recon_Alg_List = ['OSEM',
                'MLEM',
                'FBP',
                'Backprojection']

# st.write("---")
# ---- RECONSTRUCTION ----
with st.container():
    left_col, mid_col  = st.columns((1,1),gap="large")

    with left_col:
        st.subheader("PROJECTION DATA")
        # with st.expander("CHANGE PROJECTION DATA"):
        # st.subheader("Select/Upload projection data")
        # left_top_col, right_top_col = st.columns(2)
        # with left_top_col:
        sample_image = st.radio('Choose sample projection', imageNames, index=0,  horizontal=True)
        # img_path = IMAGE_DIR / sample_image
        img_path = os.path.join(IMAGE_DIR, sample_image)
        # with right_top_col:
        # uploaded_file = st.file_uploader(
        #     "or Upload projection data (.dcm or .DCM)", accept_multiple_files=False, type=['dcm', 'DCM'])
        # st.warning('Data Dimension: Projections x Slices x Bins')
        # if uploaded_file is not None:
        #     img_path = uploaded_file

        if img_path is not None:
            sino_raw = read_sino(img_path)
            
            # --- NOISE SIMULATION ---
            noise_level = st.slider("Simulate Noise (Low Counts):", 
                                  min_value=0, max_value=100, value=0, 
                                  help="Simulates lower photon counts by adding Poisson noise. 100 = very noisy (low dose), 0 = noise-free.")
            
            if noise_level > 0:
                # Scale data to simulate photon counts (lower scale = higher relative noise)
                scale = (101 - noise_level) * 10 
                sino = np.random.poisson(sino_raw * scale) / scale
            else:
                sino = sino_raw
            # -----------------------

            t,m = np.shape(sino)
            disp_img = get_disp_img(sino.T)
            
            pre, ext = os.path.splitext(img_path)
            prj_path = pre + '.png'

            st.caption('Projection')
            st.image(prj_path, width=340, clamp=True)
            st.caption('Sinogram')
            st.image(disp_img, width=340, clamp=True)


    with mid_col:
        st.subheader("IMAGE RECONSTRUCTION")
        
        selected_recon_alg = st.radio('Reconstruction Algorithm:', Recon_Alg_List, index=0)

        with st.form("Reconstruction parameter"):    
            placeholder = st.empty()
            if selected_recon_alg == Recon_Alg_List[0]:
                with placeholder.container():
                    n_ite = st.number_input('Iteration:', min_value=1, max_value=20, value=5,
                                            help="Number of full cycles through the data. More iterations improve resolution but increase 'salt-and-pepper' noise.")
                    n_subsets = st.number_input('Subset:', min_value = 1, max_value=10, value=4,
                                                help="Data is split into N subsets. Each subset update acts like a full MLEM iteration, speeding up convergence by factor of N.")
            elif selected_recon_alg ==  Recon_Alg_List[1]:
                with placeholder.container():
                    n_ite = st.number_input('Iteration:', min_value=1, max_value=20, value=10,
                                            help="Number of iterative updates. Resolution improves slowly compared to OSEM, but noise increases more predictably.")
            elif selected_recon_alg ==  Recon_Alg_List[2]: # FBP
                with placeholder.container():
                    fbp_filter = st.selectbox("Analytical Filter:", ["ramp", "shepp-logan", "cosine", "hamming", "hann"], 
                                             help="Analytical filter used to suppress 1/r blurring while balancing noise.")
                        
            submitted = st.form_submit_button("Apply")
            
        if submitted:
            # Add progress visualization placeholders
            prog_bar = st.progress(0)
            status_txt = st.empty()
            live_img = st.empty()
            
            recon_img = None
            convergence_data = []

            if selected_recon_alg == Recon_Alg_List[0]:
                recon_img, convergence_data = osem(sino, n_ite, n_subsets, prog_bar, status_txt, live_img)
                if recon_img is None:
                    st.error(f"Error: Subset count ({n_subsets}) must be a divisor of projection count ({t}).")
                recon_str = f"{selected_recon_alg} {n_ite} iter, {n_subsets} sub"
                
            elif selected_recon_alg ==  Recon_Alg_List[1]:
                recon_img, convergence_data = mlem(sino, n_ite, prog_bar, status_txt, live_img)
                recon_str = f"{selected_recon_alg} {n_ite} iter"
                
            elif selected_recon_alg ==  Recon_Alg_List[2]:
                recon_img = fbp(sino, filter_name=fbp_filter)
                recon_str = f"{selected_recon_alg} ({fbp_filter})" 
            elif selected_recon_alg ==  Recon_Alg_List[3]:
                recon_img = bp(sino)
                recon_str = selected_recon_alg
            
            if recon_img is not None:
                st.session_state.last_recon = recon_img
                st.session_state.last_recon_str = recon_str
                st.session_state.last_convergence = convergence_data
                # Clear progress indicators after completion
                prog_bar.empty()
                status_txt.empty()
                live_img.empty()

        if st.session_state.last_recon is not None:
            recon_disp_col1, recon_disp_col2 = st.columns([1, 1])
            with recon_disp_col1:
                st.write(f"**{st.session_state.last_recon_str}**")
                fig_recon = px.imshow(st.session_state.last_recon, binary_string=True)
                fig_recon.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
                fig_recon.update_layout(width=340, height=340, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_recon, use_container_width=False)

                if st.button("Add to Comparison"):
                    st.session_state.compare_image.append({
                        'name': f"{sample_image} - {st.session_state.last_recon_str}",
                        'data': st.session_state.last_recon
                    })
                    st.success(f"Added to comparison list.")

            with recon_disp_col2:
                if 'last_convergence' in st.session_state and len(st.session_state.last_convergence) > 0:
                    st.write("**Convergence Plot (RMSE)**")
                    conv_df = pd.DataFrame({
                        'Iteration': range(1, len(st.session_state.last_convergence) + 1),
                        'RMSE': st.session_state.last_convergence
                    })
                    conv_chart = alt.Chart(conv_df).mark_line(point=True).encode(
                        x='Iteration:O',
                        y=alt.Y('RMSE:Q', title='Difference from Prev Iteration'),
                        tooltip=['Iteration', 'RMSE']
                    ).properties(height=300)
                    st.altair_chart(conv_chart, use_container_width=True)
                    st.caption("Lower value indicates the algorithm is reaching a stable solution.")
                
                elif "FBP" in st.session_state.last_recon_str:
                    st.write("**Analytical Filter Shape**")
                    # Simple visualization of common FBP filters
                    f_name = st.session_state.last_recon_str.split('(')[-1].split(')')[0]
                    freq = np.linspace(0, 1, 100)
                    if f_name == 'ramp': h = freq
                    elif f_name == 'shepp-logan': h = freq * np.sinc(freq / 2)
                    elif f_name == 'cosine': h = freq * np.cos(np.pi * freq / 2)
                    elif f_name == 'hamming': h = freq * (0.54 + 0.46 * np.cos(np.pi * freq))
                    elif f_name == 'hann': h = freq * (0.5 + 0.5 * np.cos(np.pi * freq))
                    else: h = freq
                    
                    filt_df = pd.DataFrame({'Frequency': freq, 'Amplitude': h})
                    filt_chart = alt.Chart(filt_df).mark_line().encode(
                        x='Frequency',
                        y='Amplitude'
                    ).properties(height=300)
                    st.altair_chart(filt_chart, use_container_width=True)
                    st.caption(f"Frequency response of the {f_name} filter.")

    st.divider()

    # --- COMPARISON GALLERY ---
    if st.session_state.compare_image:
        st.subheader("Comparison Gallery")
        if st.button("Clear Comparison"):
            st.session_state.compare_image = []
            st.rerun()

        num_images = len(st.session_state.compare_image)
        if num_images > 0:
            titles = [img['name'] for img in st.session_state.compare_image]
            # Limit to 4 columns to avoid squishing
            cols_per_row = 4
            num_rows = (num_images + cols_per_row - 1) // cols_per_row
            
            fig_comp = make_subplots(rows=num_rows, cols=min(num_images, cols_per_row), subplot_titles=titles)
            
            for i, img_obj in enumerate(st.session_state.compare_image):
                row = (i // cols_per_row) + 1
                col = (i % cols_per_row) + 1
                fig_comp.add_trace(
                    go.Heatmap(z=img_obj['data'], colorscale='gray', showscale=False),
                    row=row, col=col
                )
            
            fig_comp.update_xaxes(showticklabels=False, matches='x')
            fig_comp.update_yaxes(
                showticklabels=False, 
                matches='y', 
                autorange='reversed',
                scaleanchor="x",
                scaleratio=1
            )
            
            fig_comp.update_layout(
                height=400 * num_rows,
                margin=dict(l=10, r=10, t=40, b=10),
                hovermode=False
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            st.info("💡 Zoom into any image to synchronize the view across all images.")
    # --------------------------

    # ---- POST-FILTER SECTION ----
    if st.session_state.last_recon is not None:
        st.divider()
        st.subheader("POST-FILTER")
        st.info("Iterative reconstructions (OSEM/MLEM) can be noisy. Applying a low-pass filter can improve clinical image quality.")
        
        filter_col1, filter_col2 = st.columns((1, 1), gap="large")
        
        with filter_col1:
            Post_Filter_list = ['None', 'Gaussian', 'Butterworth', 'Hanning']
            selected_filter = st.radio('Post-Filter type:', Post_Filter_list, index=0, horizontal=True)
            
            filt_img = st.session_state.last_recon.copy()
            filter_str = "No filter"
            shape = np.shape(filt_img)
            
            if selected_filter == 'Gaussian':
                fwhm = st.slider("FWHM (pixels):", 0.1, 10.0, 2.0)
                sigma = fwhm/2.355
                filt_img = gaussian(filt_img, sigma, preserve_range=True)
                filter_str = f"Gaussian FWHM={fwhm}"
                
            elif selected_filter == 'Butterworth':
                cutoff = st.slider("Cut-off frequency:", 0.05, 1.0, 0.25, key="bw_cutoff")
                order = st.slider("Order:", 1, 10, 5, key="bw_order")
                filt = getButterworth_lowpass_filter(shape, cutoff, order)
                filt_img = fourier_filter(filt_img, filt)
                filter_str = f"Butterworth c={cutoff}, n={order}"
                
            elif selected_filter == 'Hanning':
                cutoff = st.slider("Cut-off frequency:", 0.05, 1.0, 0.25, key="han_cutoff")
                filt = getHanning_filter(shape, cutoff)
                filt_img = fourier_filter(filt_img, filt)
                filter_str = f"Hanning c={cutoff}"

            if selected_filter != 'None':
                if st.button("Add Filtered to Comparison"):
                    st.session_state.compare_image.append({
                        'name': f"{sample_image} - {st.session_state.last_recon_str} + {filter_str}",
                        'data': filt_img
                    })
                    st.success("Filtered result added to Comparison Gallery.")

        with filter_col2:
            st.write(f"**RESULT: {filter_str}**")
            fig_filt = px.imshow(filt_img, binary_string=True)
            fig_filt.update_xaxes(showticklabels=False)
            fig_filt.update_yaxes(showticklabels=False)
            fig_filt.update_layout(width=340, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_filt, use_container_width=False)
    # -----------------------------

st.divider()
st.caption("Anucha Chaichana") 
st.caption("anucha.cha@mahidol.ac.th")
