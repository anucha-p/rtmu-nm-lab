from skimage.transform import radon, iradon
# from skimage.filters import gaussian
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
from PIL import Image

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


def mlem(sinogram, niter):
    tt = time.time()
    image_shape, nview = np.shape(sinogram)
    theta = np.linspace(0.0, 360.0, nview, endpoint=False)
    # Initial image
    mlem_rec = np.ones([image_shape, image_shape])
    # row, column = disk(
    #     (int(image_shape / 2), int(image_shape / 2)), int(image_shape / 2 - 10))
    # mlem_rec[row, column] = 1

    # Sensitivity map
    sino_ones = np.ones(sinogram.shape)
    sens_image = iradon(sino_ones, theta=theta, circle=True, filter_name=None)

    for iter in range(niter):
        # Forward projection of mlem_rec at iteration k A x^k
        fp = radon(mlem_rec, theta, circle=True)
        ratio = sinogram / (fp + 0.000001)  # ratio sinogram
        correction = iradon(ratio, theta, circle=True,
                            filter_name=None) / (sens_image+0.00000001)
        mlem_rec = mlem_rec * correction  # update

    elapsed = (time.time() - tt)
    elapsed = "{:.2f}".format(elapsed)
    st.info('Reconstruction Completed; Estimated time (sec): ' + str(elapsed))

    return mlem_rec


def osem(sinogram, niter, nsub):
    tt = time.time()
    image_shape, nview = np.shape(sinogram)
    theta = np.linspace(0.0, 360.0, nview, endpoint=False)
    # st.write(theta)
    # Initial image
    osem_rec = np.ones([image_shape, image_shape])

    # row, column = disk(
    #     (int(image_shape / 2), int(image_shape/ 2)), int(image_shape/2 - 10))
    # osem_rec[row, column] = 1
    # st.write(osem_rec.shape)
    # st.image(osem_rec, width=340)
    # Sensitivity map (Normalization matrix)
    # nview = len(theta)
    # sino_ones = np.ones(sinogram.shape)
    
    # sens_images = []
    # for sub in range(nsub):
    #     views = range(sub, nview, nsub)
    #     # st.write(views)
    #     sens_image = iradon(
    #         sino_ones[:, views], theta=theta[views], circle=True, filter_name=None)
    #     sens_images.append(sens_image)
    # st.image(sens_images[0], width=340, clamp=True)
    sino_ones = np.ones(sinogram.shape)
    sens_image = iradon(sino_ones, theta=theta, circle=True, filter_name=None)
    
    for iter in range(niter):
        order_sub = np.random.permutation(range(nsub))
        for sub in order_sub:
            views = range(sub, nview, nsub)
            # st.write(theta[views])
            # Forward projection of osem_rec at iteration k A x^k
            fp = radon(osem_rec, theta[views], circle=True)
            # st.image(fp, width=340, clamp=True)

            ratio = sinogram[:, views] / (fp + 0.000001)  # ratio sinogram
            # st.image(ratio, width=340, clamp=True)
            correction = iradon(
                ratio, theta[views], circle=True, filter_name=None) / (sens_image+0.00000001)
            # st.image(correction, width=340, clamp=True)
            # st.write(theta[views])
            osem_rec = osem_rec * correction #/ (sens_image[sub] + 0.000001) # update

    elapsed = (time.time() - tt)
    elapsed = "{:.2f}".format(elapsed)
    st.info('Reconstruction Completed; Estimated time (sec): ' + str(elapsed))
    return osem_rec


# @st.cache_data(ttl=60, max_entries=10, show_spinner="Reconstruction in progress...")
# @st.cache_data(max_entries=1)
def fbp(measured_sino, arc=360):
    tt = time.time()
    x, t = np.shape(measured_sino)
    proj_angles = np.array(range(0, arc, int(arc/t)))
    backproj = iradon(measured_sino, proj_angles, filter_name='ramp')
    elapsed = (time.time() - tt)
    elapsed = "{:.2f}".format(elapsed)
    st.info('Reconstruction Completed; Estimated time (sec): ' + str(elapsed))
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
if 'compare_filter' not in st.session_state:
    st.session_state.compare_filter = pd.DataFrame([])


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
            
            sino = read_sino(img_path)
            t,m = np.shape(sino)
            disp_img = get_disp_img(sino.T)
            
            pre, ext = os.path.splitext(img_path)
            prj_path = pre + '.png'

            st.caption('Projection')
            st.image(prj_path, width=340, clamp=True)
            # prj = np.array(Image.open(prj_path).convert('L'))
            # fig_prj = px.imshow(prj, binary_string=True)
            # fig_prj.update_xaxes(showticklabels=False)
            # fig_prj.update_yaxes(showticklabels=False)
            # fig_prj.update_layout(title_text="Projection")
            # st.plotly_chart(fig_prj, use_container_width=True)
            st.caption('Sinogram')
            st.image(disp_img, width=340, clamp=True)
            # fig_sino = px.imshow(sino.T, binary_string=True)
            # fig_sino.update_xaxes(showticklabels=False)
            # fig_sino.update_yaxes(showticklabels=False)
            # fig_sino.update_layout(title_text="Sinogram")
            # st.plotly_chart(fig_sino, use_container_width=True)


    with mid_col:
        st.subheader("IMAGE RECONSTRUCTION")
        
        selected_recon_alg = st.radio('Reconstruction Algorithm:', Recon_Alg_List, index=0)

        with st.form("Filter parameter"):    
            # sino = img[:,slice_loc,:].copy()
            # sino = np.swapaxes(sino,0,1)
            placeholder = st.empty()
            recon_img = None
            if selected_recon_alg == Recon_Alg_List[0]:
                with placeholder.container():
                    # rcol1, rcol2 = st.columns(2)
                    # with rcol1:
                    n_ite = st.number_input('Number of iteration:', min_value=1)
                    # with rcol2:
                    n_subsets = st.number_input('Number of subsets:', min_value = 1, help="Number of subsets should be a divisor of the total number of projections.")
            elif selected_recon_alg ==  Recon_Alg_List[1]:
                with placeholder.container():
                    n_ite = st.number_input('Number of iteration:', min_value=1)
                        

            # elapsed = (time.time() - t)*m
            # elapsed = "{:.2f}".format(elapsed)
            # st.write('Estimated reconstruction time (sec): ' + str(elapsed))
            
            
            submitted = st.form_submit_button("Apply")
            if submitted:
                if selected_recon_alg == Recon_Alg_List[0]:
                    recon_img = osem(sino, n_ite, n_subsets)
                    if recon_img is None:
                        st.caption('#subsets ('+str(n_subsets) + ') is not a divisor of #projections (' + str(t) +')')
                    recon_str = selected_recon_alg +' '+str(n_ite)+' iteration, '+str(n_subsets)+' subset'
                    
                elif selected_recon_alg ==  Recon_Alg_List[1]:
                    recon_img = mlem(sino, n_ite)
                    recon_str = selected_recon_alg +' '+str(n_ite)+' iteration'
                    
                elif selected_recon_alg ==  Recon_Alg_List[2]:
                    recon_img = fbp(sino)
                    recon_str = selected_recon_alg 
                elif selected_recon_alg ==  Recon_Alg_List[3]:
                    recon_img = bp(sino)
                    recon_str = selected_recon_alg  
                    
            if recon_img is not None:
                # st.write("RECONSTRUCTED IMAGE")
                # disp_recon_img = (np.maximum(recon_img, 0) /
                #                 recon_img.max()) * 255.0
                # disp_recon_img = np.uint8(disp_recon_img)
                # st.image(recon_img, width=340)
                st.write(recon_str)
                fig_recon = px.imshow(recon_img, binary_string=True)
                fig_recon.update_xaxes(showticklabels=False)
                fig_recon.update_yaxes(showticklabels=False)
                fig_recon.update_layout(width=340, title_text="RECONSTRUCTED IMAGE")
                st.plotly_chart(fig_recon, use_container_width=False)

    st.divider()
# with st.container():
#     st.subheader("POST-FILTER")
#     mid_col2, right_col = st.columns((1, 1), gap="large")
#     with mid_col2:
#         selected_filter = st.selectbox('Filter type:', Filter_list, index=0)
            
#         placeholder2 = st.empty()
#         if selected_filter == Filter_list[0]:
#             filt_img = recon_img
#             filter_str=''
        
#         elif selected_filter == Filter_list[1]:
#             with placeholder2.container():
#                 fwhm = st.slider("Full-width at half maximum (pixel):", min_value=1.0, max_value=10.0, step=0.1, value=2.0)
#                 sigma = fwhm/(2.355)
#                 truncate = 4
#                 radius = int(truncate * sigma + 0.5)
#                 ksize = 2 * radius + 1
#                 filt_img = gaussian(img, sigma, truncate=truncate, preserve_range=True, mode='reflect')
#                 gauss_ker = gaussianKernel2(ksize, sigma,  twoDimensional=False)

#                 filter_str = selected_filter + '/' + str(fwhm)
            
#         elif selected_filter ==  Filter_list[2]:
#             with placeholder2.container():
#                 cutoff = st.slider("Cut-off:", min_value=0.01, max_value=1.0, step=0.01, value=0.25)
#                 order = st.slider("Order:", min_value=2, max_value=10, step=1)
#                 shape = np.shape(recon_img)
#                 filt = getButterworth_lowpass_filter(shape, cutoff, order)
#                 filter_str = selected_filter + '/' + str(cutoff) + '/' + str(order)

#         elif selected_filter ==  Filter_list[3]:
#             with placeholder2.container():
#                 cutoff = st.slider("Cut-off:", min_value=0.01, max_value=1.0, step=0.01, value=0.25)
#                 shape = np.shape(recon_img)
#                 filt = getHanning_filter(shape, cutoff)
#                 filter_str = selected_filter + '/' + str(cutoff) 

#         elif selected_filter ==  Filter_list[4]:
#             with placeholder2.container():
#                 cutoff = st.slider("Cut-off:", min_value=0.01, max_value=1.0, step=0.01, value=0.25)
#                 shape = np.shape(recon_img)
#                 filt = getHamming_filter(shape, cutoff)
#                 filter_str = selected_filter + '/' + str(cutoff) 


#         if selected_filter == Filter_list[0]:
#             pass
#         elif selected_filter == Filter_list[1]:
#             df = pd.DataFrame({
#                 'x':range(len(gauss_ker)),
#                 'y':gauss_ker})
#             line_chart = alt.Chart(df).mark_line(interpolate='basis').encode(
#                 alt.X('x', title='Pixels'),
#                 alt.Y('y', title='Weight')
#             )
#             with st.container():
#                 st.altair_chart(line_chart, use_container_width=True)
#         else:
#             filt_img =  fourier_filter(recon_img, filt)
#             df = pd.DataFrame({
#                 'x':x,
#                 'y':filt[int(shape[0]/ 2), int(shape[1]/ 2):shape[1]]
#             })

#             line_chart = alt.Chart(df).mark_line(interpolate='basis').encode(
#                 alt.X('x', title='Frequency (cycle/pixel)'),
#                 alt.Y('y', title='Amplitude')
#             )
#             with st.container():
#                 st.altair_chart(line_chart, use_container_width=True)


#     with right_col:
#         st.write("FILTERED IMAGE")
        
#         if recon_img is not None:
#             disp_recon_img = (np.maximum(filt_img, 0) / filt_img.max()) * 255.0
#             disp_recon_img = np.uint8(disp_recon_img)
#             st.image(disp_recon_img, width=340)

#         st.write(recon_str)
#         st.write(filter_str)

# st.divider()
st.caption("Anucha Chaichana") 
st.caption("anucha.cha@mahidol.edu")
