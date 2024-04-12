from PIL import Image
# import requests
import streamlit as st
# from streamlit_image_comparison import image_comparison
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# import cv2
import pydicom as dicom
import math
# import os
import pandas as pd
import altair as alt
from skimage.filters import gaussian
from scipy.interpolate import make_interp_spline

from src.Filters import *
# import random

st.set_page_config(page_title="Filter", page_icon="✋🏻", layout="wide")

remove_top_padding = """
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
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


# @st.cache_data(max_entries=1)
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

# @st.cache_data(persist="disk")
def gaussianKernel2(size, sigma, twoDimensional=True):
    """
    Creates a gaussian kernel with given sigma and size, 3rd argument is for choose the kernel as 1d or 2d
    """
    if twoDimensional:
        kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    else:
        kernel = np.fromfunction(lambda x: math.e ** ((-1*(x-(size-1)/2)**2) / (2*sigma**2)), (size,))
    return kernel / np.sum(kernel)

# @st.cache_data(persist="disk")
# def display_org(org_img):
    

# @st.cache_data(max_entries=1)
def read_image(img_path):
    if img_path.name.endswith('.dcm') or img_path.name.endswith('.DCM'):
        ds = dicom.dcmread(img_path)
        img = ds.pixel_array.astype(float)
        if img.ndim > 2:
            s = img.shape[0]
            img = img[int(s/2), :, :]
        scaled_image = (np.maximum(img, 0) / img.max()) * 255.0
        img = np.uint8(scaled_image)
    else:
        img = np.array(Image.open(img_path).convert('L'))

    m, n = np.shape(img)
    # shape = np.shape(img)

    if m > n:
        img = np.pad(img, ((0, 0), (0, int(m-n))))
    elif m < n:
        img = np.pad(img, ((0, int(n-m)), (0, 0)))

    return img

# @st.cache_data(persist="disk")
# def apply_filter(image, filter):

    


# ---- HEADER SECTION ----
with st.container():
    st.title("Image Filtering in Nuclear Medicine")
    st.write("---")


# ---- LOAD IMAGE ----
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / 'images/filter' 
imageNames = [f.name for f in IMAGE_DIR.iterdir()]

st.sidebar.header("Filter")

# with st.sidebar:
# with st.expander("CHANGE INPUT IMAGE"):
left_top_col, right_top_col  = st.columns((1,1))
with left_top_col:
    
    sample_image = st.radio('Choose sample image', imageNames, index=0)
    
with right_top_col:

    uploaded_file = st.file_uploader("or Upload image", accept_multiple_files=False, type=['dcm','DCM','jpg','JPG','jpeg','png'])
    if uploaded_file is not None:
        img_path = uploaded_file
        st.write("filename:", uploaded_file.name)
    else:
        img_path = IMAGE_DIR / sample_image


img = read_image(img_path)
m,n = np.shape(img)
shape = np.shape(img)
x = np.linspace(0,1,int(m/2))
filt_img = None

# ---- FILTER ----
with st.container():
    # st.write("---")
    
    Filter_list = ['Gaussian', 
            'Butterworth', 
            'Hanning']

    left_col, right_col  = st.columns(2, gap="large")

    with left_col:
        st.subheader('Filter')
        # selected_filter = st.selectbox('Filter type:', Filter_list, index=1)
        selected_filter = st.radio("Filter type:", Filter_list)

        with st.form("Filter parameter"):
            
            placeholder = st.empty()
            if selected_filter == Filter_list[0]:
                with placeholder.container():               
                    fwhm = st.slider("Full-width at half maximum (pixel):", min_value=1.0, max_value=21.0, step=2.0, value=3.0)
            elif selected_filter ==  Filter_list[1]:
                with placeholder.container():
                    cutoff = st.slider("Cut-off:", min_value=0.05, max_value=1.0, step=0.05, value=0.5)
                    order = st.slider("Order:", min_value=1, max_value=10, step=1)
                    
            else:
                with placeholder.container():
                    cutoff = st.slider("Cut-off:", min_value=0.05, max_value=1.0, step=0.05, value=0.5)
                    

            # elif selected_filter ==  Filter_list[3]:
            #     with placeholder.container():
            #         cutoff = st.slider("Cut-off:", min_value=0.01, max_value=1.0, step=0.05, value=0.25)
            #         filt = getHamming_filter(shape, cutoff)

            submitted = st.form_submit_button("Apply")
            if submitted:

                if selected_filter == Filter_list[0]:
                    sigma = fwhm/(2.355)
                    truncate = 4
                    radius = int(truncate * sigma + 0.5)
                    ksize = 2 * radius + 1
                    filt_img = gaussian(img, sigma, truncate=truncate, preserve_range=True, mode='reflect')
                    gauss_ker = gaussianKernel2(ksize, sigma,  twoDimensional=False)
                    # st.line_chart(gauss_ker)
                    df = pd.DataFrame({
                        'x':range(len(gauss_ker)),
                        'y':gauss_ker})
                    line_chart = alt.Chart(df).mark_line(interpolate='basis').encode(
                        alt.X('x', title='Pixels'),
                        alt.Y('y', title='Kernel weight')
                    )
                    with st.container():
                        st.altair_chart(line_chart, use_container_width=True)
                else:
                    if selected_filter ==  Filter_list[1]:
                        filt = getButterworth_lowpass_filter(shape, cutoff, order)
                    else:   
                        filt = getHanning_filter(shape, cutoff)
                        
                    filt_img =  fourier_filter(img, filt)
                    df = pd.DataFrame({
                            'x':x,
                            'y':filt[int(shape[0]/ 2), int(shape[1]/ 2):shape[1]]
                        })

                    line_chart = alt.Chart(df).mark_line(interpolate='basis').encode(
                        alt.X('x', title='Frequency (cycle/pixel)'),
                        alt.Y('y', title='Amplitude')
                    )
                    with st.container():
                        st.altair_chart(line_chart, use_container_width=True)

    with right_col:

        # st.write("##")
        # @st.cache_data(persist="disk")
        def display_org(img):
            org_img = Image.fromarray(img)
            st.subheader('Original Image')
            st.image(org_img, width=340)
        display_org(img)
        
        if filt_img is not None:
            filt_img = filt_img[0:m, 0:n]
            img_disp = Image.fromarray(filt_img)
            img_disp = img_disp.convert("L")
            st.subheader('Filtered Image')
            st.image(img_disp, width=340)
        
            if selected_filter == Filter_list[0]:
                st.write(selected_filter, 'FWHM', fwhm)
            elif selected_filter == Filter_list[1]:
                st.write(selected_filter, 'with Cut-off:', cutoff, ' Order:', order)
            else:
                st.write(selected_filter, 'with Cut-off:', cutoff)

st.write("---")


# if filt_img is not None:
display = st.checkbox('SEE FILTERING PROCESS')

if display and (filt_img is not None):

    with st.container():
        if selected_filter == Filter_list[0]:
            # with st.expander('SEE FILTERING PROCESS'):
                # gauss_2D = np.sqrt(gauss_ker * np.transpose(gauss_ker))
                gauss_2D = gaussianKernel2(ksize, sigma)
                fig_1, axr_1 = plt.subplots(3,2, figsize = (10,10))
                font_size = 8
                axr_1[0,0].imshow(img, cmap='gray')
                axr_1[0,0].set_title('Original image', fontsize=font_size)
                axr_1[0,1].plot(img[int(shape[0]/ 2),:])
                axr_1[0,1].set_title('Central profile of the original image', fontsize=font_size)

                axr_1[1,0].imshow(gauss_2D, cmap='gray')
                axr_1[1,0].set_title('2D Gaussian kernel size= '+str(ksize)+"x"+str(ksize), fontsize=font_size)
                
                axr_1[1,1].plot(gauss_ker)
                axr_1[1,1].set_title('1D Gaussian kernel', fontsize=font_size)

                axr_1[2,0].imshow(filt_img, cmap='gray')
                axr_1[2,0].set_title('Filtered image', fontsize=font_size)
                axr_1[2,1].plot(filt_img[int(shape[0]/ 2),:])
                axr_1[2,1].set_title('Central profile of the filtered image', fontsize=font_size)
                
                st.pyplot(fig_1)

        else:
            # with st.expander('SEE FITERING PROCESS'):
                intro_markdown = read_markdown_file(BASE_DIR / "markdown/Text directory.md")
                st.markdown(intro_markdown, unsafe_allow_html=True)

                image_fft = np.fft.fft2(img)
                shift_fft = np.fft.fftshift(image_fft)
                mag_img_dft = np.log(np.abs(shift_fft)+1)
                filt_dft = np.multiply(filt, shift_fft)
                mag_filt_dft = np.log(np.abs(filt_dft)+1)
                mag_filt= np.log(np.abs(filt)+1)

                fig, axr = plt.subplots(3,3, figsize = (10,10))
                font_size = 8
                axr[0,0].imshow(img, cmap='gray')
                axr[0,0].set_title('Original image, f', fontsize=font_size)
                axr[0,1].imshow(mag_img_dft, cmap='gray')
                axr[0,1].set_title('Image in Frequency Domain, F', fontsize=font_size)
                axr[0,0].set_xticklabels('')
                axr[0,1].set_xticklabels('')
                axr[0,0].set_yticklabels('')
                axr[0,1].set_yticklabels('')
                axr[0,2].plot(x, mag_img_dft[int(shape[0]/ 2),int(shape[1]/ 2):shape[1]])

                axr[1,1].imshow(mag_filt, cmap='gray')
                if selected_filter == 1:
                    axr[1,1].set_title(selected_filter +' G with Cut-off = ' + str(cutoff) + ', Order = ' + str(order), fontsize=font_size)
                else:
                    axr[1,1].set_title(selected_filter +' G with Cut-off = ' + str(cutoff), fontsize=font_size)
                axr[1,2].plot(x,filt[int(shape[0] / 2),int(shape[1]/ 2):shape[1]])
                axr[1,0].set_xticklabels('')
                axr[1,1].set_xticklabels('')
                axr[1,0].set_yticklabels('')
                axr[1,1].set_yticklabels('')

                axr[2,0].imshow(filt_img, cmap='gray')
                axr[2,0].set_title('Filtered image', fontsize=font_size)
                axr[2,1].imshow(mag_filt_dft, cmap='gray')
                axr[2,1].set_title('Filtered image F*G in Frequency Domain', fontsize=font_size)
                axr[2,0].set_xticklabels('')
                axr[2,1].set_xticklabels('')
                axr[2,0].set_yticklabels('')
                axr[2,1].set_yticklabels('')
                axr[2,2].plot(x, mag_filt_dft[int(shape[0]/ 2),int(shape[1]/ 2):shape[1]])

                st.pyplot(fig)

    
st.write("---")
st.caption("Anucha Chaichana") 
st.caption("anucha.cha@mahidol.edu")
