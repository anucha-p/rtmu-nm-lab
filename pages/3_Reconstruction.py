from skimage.transform import radon, iradon
from skimage.filters import gaussian
from skimage.draw import disk
import streamlit as st
from pathlib import Path

import numpy as np
import pydicom as dicom
import math
import pandas as pd
import altair as alt
import streamlit_nested_layout
import time


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
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / 'images/recon' 
imageNames = [f.name for f in IMAGE_DIR.iterdir() if f.suffix=='.dcm']

st.sidebar.header("Reconstruction")


def gaussianKernel2(size, sigma, twoDimensional=True):
    """
    Creates a gaussian kernel with given sigma and size, 3rd argument is for choose the kernel as 1d or 2d
    """
    if twoDimensional:
        kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    else:
        kernel = np.fromfunction(lambda x: math.e ** ((-1*(x-(size-1)/2)**2) / (2*sigma**2)), (size,))
    return kernel / np.sum(kernel)


def mlem(sinogram, niter):
    tt = time.time()
    image_shape, nview = np.shape(sinogram)
    theta = np.linspace(0.0, 360.0, nview, endpoint=False)
    # Initial image
    mlem_rec = np.ones([image_shape, image_shape])
    row, column = disk(
        (int(image_shape / 2), int(image_shape / 2)), int(image_shape / 2 - 10))
    mlem_rec[row, column] = 1

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
    osem_rec = np.zeros([image_shape, image_shape])*100

    row, column = disk(
        (int(image_shape / 2), int(image_shape/ 2)), int(image_shape/2 - 10))
    osem_rec[row, column] = 1
    # st.write(osem_rec.shape)
    # st.image(osem_rec, width=340)
    # Sensitivity map (Normalization matrix)
    # nview = len(theta)
    sino_ones = np.ones(sinogram.shape)
    
    sens_images = []
    for sub in range(nsub):
        views = range(sub, nview, nsub)
        # st.write(views)
        sens_image = iradon(
            sino_ones[:, views], theta=theta[views], circle=True, filter_name=None)
        sens_images.append(sens_image)
    # st.image(sens_images[0], width=340, clamp=True)
    
    
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
                ratio, theta[views], circle=True, filter_name=None) #/ (sens_image[sub] + 0.000001)
            # st.image(correction, width=340, clamp=True)
            # st.write(theta[views])
            osem_rec = osem_rec * correction  # update

    elapsed = (time.time() - tt)
    elapsed = "{:.2f}".format(elapsed)
    st.info('Reconstruction Completed; Estimated time (sec): ' + str(elapsed))
    return osem_rec


# @st.cache_data(ttl=60, max_entries=10, show_spinner="Reconstruction in progress...")
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
def bp(measured_sino, arc=360):
    tt = time.time()
    x, t = np.shape(measured_sino)
    proj_angles = np.array(range(0, arc, int(arc/t)))
    backproj = iradon(measured_sino, proj_angles, filter_name=None)
    elapsed = (time.time() - tt)
    elapsed = "{:.2f}".format(elapsed)
    st.info('Reconstruction Completed; Estimated time (sec): ' + str(elapsed))
    return backproj


# @st.cache_data
def read_dcm(dcm_img):
    ds = dicom.dcmread(dcm_img)
    img = ds.pixel_array.astype(float)
    return img


# @st.cache_data
def get_disp_img(img):
    scaled_image = (np.maximum(img, 0) / img.max()) * 255.0
    disp_img = np.uint8(scaled_image)
    return disp_img


# @st.cache_data(ttl=60, max_entries=10, show_spinner="Reconstruction in progress...")
def getButterworth_lowpass_filter(shape, cutoff=0.25, order=2):
    m, n = shape
    d0 = cutoff
    h = np.zeros((m, n))
    X = np.linspace(-1, 1, shape[0])
    Y = np.linspace(-1, 1, shape[0])
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            d = math.sqrt((x ** 2) + (y ** 2))
            h[i, j] = 1 / (1 + (d / d0) ** (2 * order))
    return h


# @st.cache_data(ttl=60, max_entries=10, show_spinner="Reconstruction in progress...")
def getButterworth_highpass_filter(shape, cutoff=0.25, order=2):
    m, n = shape
    d0 = cutoff
    h = np.zeros((m, n))
    X = np.linspace(-1, 1, shape[0])
    Y = np.linspace(-1, 1, shape[0])
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            d = math.sqrt((x ** 2) + (y ** 2))
            h[i, j] = 1 / (1 + (d0 / d) ** (2 * order))
    return h


# @st.cache_data
def getHanning_filter(shape, cutoff=0.25):
    m, n = shape
    d0 = cutoff
    h = np.zeros((m, n))
    X = np.linspace(-1, 1, shape[0])
    Y = np.linspace(-1, 1, shape[0])
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            d = math.sqrt((x ** 2) + (y ** 2))
            if 0 <= d and d <= d0:
                h[i, j] = 0.5 + 0.5*math.cos(math.pi*d/d0)
            else:
                h[i, j] = 0
    return h


# @st.cache_data
def getHamming_filter(shape, cutoff=0.25):
    m, n = shape
    d0 = cutoff
    h = np.zeros((m, n))
    X = np.linspace(-1, 1, shape[0])
    Y = np.linspace(-1, 1, shape[0])
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            d = math.sqrt((x ** 2) + (y ** 2))
            if 0 <= d and d <= d0:
                h[i, j] = 0.54 + 0.46*math.cos(math.pi*d/d0)
            else:
                h[i, j] = 0
    return h


# @st.cache_data
def fourier_filter(image, filt):
    image_fft = np.fft.fft2(image)
    shift_fft = np.fft.fftshift(image_fft)
    filtered_image = np.multiply(filt, shift_fft)
    shift_ifft = np.fft.ifftshift(filtered_image)
    ifft = np.fft.ifft2(shift_ifft)
    filt_image = np.abs(ifft)
    return filt_image


# @st.cache_data
def getGaussion_filter(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


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

Filter_list = ['None',
'Gaussian', 
'Butterworth', 
'Hanning',
'Hamming']

Recon_Alg_List = ['OSEM',
                'MLEM',
                'FBP',
                'Simple BP']


# st.write("---")
# ---- RECONSTRUCTION ----
with st.container():
    left_col, mid_col  = st.columns((1,1),gap="large")

    with left_col:
        st.subheader("PROJECTION DATA")
        with st.expander("CHANGE PROJECTION DATA"):
            # st.subheader("Select/Upload projection data")
            # left_top_col, right_top_col = st.columns(2)
            # with left_top_col:
            sample_image = st.selectbox('Choose sample projection', imageNames, index=0)
            img_path = IMAGE_DIR / sample_image
            # with right_top_col:
            uploaded_file = st.file_uploader(
                "or Upload projection data (.dcm or .DCM)", accept_multiple_files=False, type=['dcm', 'DCM'])
            st.warning('Data Dimension: Projections x Slices x Bins')
            if uploaded_file is not None:
                img_path = uploaded_file

        if img_path is not None:
            img = read_dcm(img_path)
            disp_img = get_disp_img(img)
            t, m, n = np.shape(img)
            x = np.linspace(0, 1, int(m/2))

            # st.write('Num of Projection: ' + str(t) + ',  Num of Slice: '+ str(m)+',  Num of Bin: '+ str(n))
            angle = st.slider("Projection:", min_value=1, max_value=t, step=1, value=1)
            slice_loc = st.slider("Slice:", min_value=1, max_value=m, step=1, value=int(m/2))
            proj_angle = np.array(range(0,360,int(360/t)))

            proj_img = disp_img[angle,:,:].copy()
            proj_img[slice_loc,:] = 255.0
            st.image(proj_img, width=340)


    with mid_col:
        st.subheader("IMAGE RECONSTRUCTION")
        selected_recon_alg = st.selectbox('Reconstruction Algorithm:', Recon_Alg_List, index=0)

        
        sino = img[:,slice_loc,:].copy()
        sino = np.swapaxes(sino,0,1)
        placeholder = st.empty()
        recon_img = None
        if selected_recon_alg == Recon_Alg_List[0]:
            with placeholder.container():
                rcol1, rcol2 = st.columns(2)
                with rcol1:
                    n_ite = st.number_input('Number of iteration:', min_value=1)
                with rcol2:
                    n_subsets = st.number_input('Number of subsets:', min_value = 1, help="Number of subsets should be a divisor of the total number of projections.")
                # t = time.time()
                if st.button('Start Recon'):
                    # recon_img = osem(sino, n_ite, n_subsets)
                    recon_img = osem(sino, n_ite, n_subsets)
                    if recon_img is None:
                        st.caption('#subsets ('+str(n_subsets) + ') is not a divisor of #projections (' + str(t) +')')
                
                recon_str = selected_recon_alg +' ('+str(n_ite)+'x'+str(n_subsets)+')'

        elif selected_recon_alg ==  Recon_Alg_List[1]:
            with placeholder.container():
                n_ite = st.number_input('Number of iteration:', min_value=1)
                # t = time.time()
                if st.button('Start Recon'):
                    recon_img = mlem(sino, n_ite)
                recon_str = selected_recon_alg +' ('+str(n_ite)+')'

        elif selected_recon_alg ==  Recon_Alg_List[2]:
            with placeholder.container():
                # t = time.time()
                if st.button('Start Recon'):
                    recon_img = fbp(sino)
                recon_str = selected_recon_alg 

        elif selected_recon_alg ==  Recon_Alg_List[3]:
            with placeholder.container():
                # t = time.time()
                if st.button('Start Recon'):
                    recon_img = bp(sino)
                recon_str = selected_recon_alg
        # elapsed = (time.time() - t)*m
        # elapsed = "{:.2f}".format(elapsed)
        # st.write('Estimated reconstruction time (sec): ' + str(elapsed))
        if recon_img is not None:
            st.caption("RECONSTRUCTED IMAGE")
            disp_recon_img = (np.maximum(recon_img, 0) /
                            recon_img.max()) * 255.0
            disp_recon_img = np.uint8(disp_recon_img)
            st.image(disp_recon_img, width=340)

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
