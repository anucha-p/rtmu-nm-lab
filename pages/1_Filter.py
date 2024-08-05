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
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly import colors
import os

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
# with st.container():
st.title("Image Filtering in Nuclear Medicine")
st.write("---")


# ---- LOAD IMAGE ----
# BASE_DIR = Path(__file__).resolve().parent
# IMAGE_DIR = BASE_DIR / 'images/filter' 
# imageNames = [f.name for f in IMAGE_DIR.iterdir()]

BASE_DIR = os.path.abspath(os.path.join(__file__, '../'))
IMAGE_DIR = os.path.join(BASE_DIR, 'images/filter')
file_name = os.listdir(IMAGE_DIR)
imageNames = [f for f in file_name]

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
        # st.write("filename:", uploaded_file.name)
    else:
        img_path = os.path.join(IMAGE_DIR, sample_image)
        # img_path = IMAGE_DIR / sample_image

st.write("---")

img = read_image(img_path)
m,n = np.shape(img)
shape = np.shape(img)
x = np.linspace(0,1,int(m/2))
filt_img = None

# ---- FILTER ----
# with st.container():
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
                order = st.slider("Order:", min_value=1, max_value=10, step=1, value=3)
                
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
                
                # gauss_2D = gaussianKernel2(ksize, sigma)
                # fig = go.Figure()
                # trace=go.Scatter(x=np.linspace(0,ksize,ksize, endpoint=False),
                #     y=np.squeeze(gauss_2D[int(ksize/2),:]),
                #     line=dict(width=2),
                #     showlegend=False)
                # fig.add_trace(trace)
                # fig.update_layout(title='Gaussian Kernel',
                #    xaxis_title='Pixe;',
                #    yaxis_title='Kernel weight')
                # st.plotly_chart(fig, use_container_width=True)
                
                # g_kernel = px.imshow(gauss_2D, binary_string=True)
                # g_kernel.update_xaxes(showticklabels=False)
                # g_kernel.update_yaxes(showticklabels=False)
                # g_kernel.update_layout(width=100)
                # st.plotly_chart(g_kernel, use_container_width=True)
                
                
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
    # st.subheader('Original Image')
    # st.write("##")
    # @st.cache_data(persist="disk")
    # def display_org(img):
    #     org_img = Image.fromarray(img)
    #     st.subheader('Original Image')
    #     # st.image(org_img, width=340)
    # display_org(img)
    
    # st.image(org_img, width=340)
    fig_org = px.imshow(img, binary_string=True)
    fig_org.update_xaxes(showticklabels=False)
    fig_org.update_yaxes(showticklabels=False)
    fig_org.update_layout(coloraxis_showscale=False)
    fig_org.update_layout(width=340, title_text="Original Image")
    st.plotly_chart(fig_org, use_container_width=False)
   
    
    if filt_img is not None:
        filt_img = filt_img[0:m, 0:n]
        # img_disp = Image.fromarray(filt_img)
        # img_disp = img_disp.convert("L")
        # st.subheader('Filtered Image')
        # st.image(img_disp, width=340)
        # fig_org = px.imshow(img, binary_string=True)
        # fig_org.update_layout(width=340, height=340)
        # st.plotly_chart(fig_org, use_container_width=True)
        
        fig_filt = px.imshow(filt_img, binary_string=True)
        fig_filt.update_xaxes(showticklabels=False)
        fig_filt.update_yaxes(showticklabels=False)
        fig_filt.update_layout(width=340, title_text="Filtered Image")
        st.plotly_chart(fig_filt, use_container_width=False)
        
        # fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.0025)
        # fig.add_trace(fig_org.data[0], 1, 1)
        # fig.add_trace(fig_filt.data[0], 2, 1)
        # fig.update_xaxes(showticklabels=False)
        # fig.update_yaxes(showticklabels=False)
        # fig.update_layout(coloraxis_showscale=False)
        # fig.update_layout(width=340, height=700)
        # st.plotly_chart(fig, use_container_width=True)
    
        if selected_filter == Filter_list[0]:
            st.write(selected_filter, 'FWHM', fwhm)
        elif selected_filter == Filter_list[1]:
            st.write(selected_filter, 'with Cut-off:', cutoff, ' Order:', order)
        else:
            st.write(selected_filter, 'with Cut-off:', cutoff)
    # else:
        
        
st.write("---")


# if filt_img is not None:
display = st.checkbox('More details')

if display and (filt_img is not None):
    cols = colors.DEFAULT_PLOTLY_COLORS
    # with st.container():
    if selected_filter == Filter_list[0]:
        gauss_2D = gaussianKernel2(ksize, sigma)
        # # with st.expander('SEE FILTERING PROCESS'):
        #     # gauss_2D = np.sqrt(gauss_ker * np.transpose(gauss_ker))
        #     gauss_2D = gaussianKernel2(ksize, sigma)
        #     fig_1, axr_1 = plt.subplots(3,2, figsize = (10,10))
        #     font_size = 8
        #     axr_1[0,0].imshow(img, cmap='gray')
        #     axr_1[0,0].set_title('Original image', fontsize=font_size)
        #     axr_1[0,1].plot(img[int(shape[0]/ 2),:])
        #     axr_1[0,1].set_title('Central profile of the original image', fontsize=font_size)

        #     axr_1[1,0].imshow(gauss_2D, cmap='gray')
        #     axr_1[1,0].set_title('2D Gaussian kernel size= '+str(ksize)+"x"+str(ksize), fontsize=font_size)
            
        #     axr_1[1,1].plot(gauss_ker)
        #     axr_1[1,1].set_title('1D Gaussian kernel', fontsize=font_size)

        #     axr_1[2,0].imshow(filt_img, cmap='gray')
        #     axr_1[2,0].set_title('Filtered image', fontsize=font_size)
        #     axr_1[2,1].plot(filt_img[int(shape[0]/ 2),:])
        #     axr_1[2,1].set_title('Central profile of the filtered image', fontsize=font_size)
            
        #     st.pyplot(fig_1)
        # figm = px.imshow(img, binary_string=True)
        # figm.update_layout(width=400, height=400)
        
        # fig_filt= px.imshow(filt_img, binary_string=True)

        
        fig = make_subplots(rows=4, cols=5,
                            specs=[[{"rowspan": 2, "colspan": 2}, None, {"rowspan": 2, "colspan": 2}, None, {}],
                                [None, None, None, None, {}],
                                [{"rowspan": 2, "colspan": 4}, None, None, None, {}],
                                [None, None, None, None, {}]],
                            shared_xaxes=True, 
                            vertical_spacing=0.0025,
                            horizontal_spacing=0.0025,
                            subplot_titles=['Original image','Filtered image'])
        fig.add_trace(fig_org.data[0], 1, 1)
        trace=go.Scatter(x=np.linspace(0,n,n, endpoint=False),
                 y=np.squeeze(img[int(shape[0]/ 2),:]),
                 line=dict(width=2),
                 showlegend=False)
        fig.add_trace(trace, 3, 1)
        # fig.add_trace(go.Scatter(x=np.linspace(0,n,n, endpoint=False), y=np.squeeze(img[int(shape[0]/ 2),:])), 2, 1,)
        fig.add_trace(fig_filt.data[0], 1, 3)
        trace=go.Scatter(x=np.linspace(0,n,n, endpoint=False),
                 y=np.squeeze(filt_img[int(shape[0]/ 2),:]),
                 line=dict(width=2),
                 showlegend=False)
        fig.add_trace(trace, 3, 1)
        # fig.add_trace(go.Scatter(x=np.linspace(0,n,n, endpoint=False), y=np.squeeze(filt_img[int(shape[0]/ 2),:])), 2, 1)
        
        gauss_2D = gaussianKernel2(ksize, sigma)
        # fig = go.Figure()
        trace=go.Scatter(x=np.linspace(0,ksize,ksize, endpoint=False),
            y=np.squeeze(gauss_2D[int(ksize/2),:]),
            line=dict(width=2),
            showlegend=False)
        fig.add_trace(trace, 1, 5)
        fig.update_layout(title='Gaussian Kernel',
           xaxis_title='Pixe;',
           yaxis_title='Kernel weight')
        # st.plotly_chart(fig, use_container_width=True)
        
        g_kernel = px.imshow(gauss_2D, binary_string=True)
        g_kernel.update_xaxes(showticklabels=False)
        g_kernel.update_yaxes(showticklabels=False)
        g_kernel.update_layout(width=100)
        fig.add_trace(g_kernel.data[0], 2, 5)
        # st.plotly_chart(g_kernel, use_container_width=True)
        
        # fig.show()
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(width=700, height=800)
        st.plotly_chart(fig, use_container_width=False)
        # g_left_col, g_right_col  = st.columns(2, gap="large")
        # with g_left_col:
        #     st.plotly_chart(fig_org, use_container_width=False)
            

    else:
        # with st.expander('SEE FITERING PROCESS'):
            # intro_markdown = read_markdown_file(BASE_DIR / "markdown/Text directory.md")
            intro_markdown = read_markdown_file(os.path.join(BASE_DIR, "markdown/Text directory.md"))
            st.markdown(intro_markdown, unsafe_allow_html=True)

            image_fft = np.fft.fft2(img)
            shift_fft = np.fft.fftshift(image_fft)
            mag_img_dft = np.log(np.abs(shift_fft)+1)
            filt_dft = np.multiply(filt, shift_fft)
            mag_filt_dft = np.log(np.abs(filt_dft)+1)
            mag_filt= np.log(np.abs(filt)+1)

            # fig, axr = plt.subplots(3,3, figsize = (10,10))
            # font_size = 8
            # axr[0,0].imshow(img, cmap='gray')
            # axr[0,0].set_title('Original image, f', fontsize=font_size)
            # axr[0,1].imshow(mag_img_dft, cmap='gray')
            # axr[0,1].set_title('Image in Frequency Domain, F', fontsize=font_size)
            # axr[0,0].set_xticklabels('')
            # axr[0,1].set_xticklabels('')
            # axr[0,0].set_yticklabels('')
            # axr[0,1].set_yticklabels('')
            # axr[0,2].plot(x, mag_img_dft[int(shape[0]/ 2),int(shape[1]/ 2):shape[1]])

            # axr[1,1].imshow(mag_filt, cmap='gray')
            # if selected_filter == 1:
            #     axr[1,1].set_title(selected_filter +' G with Cut-off = ' + str(cutoff) + ', Order = ' + str(order), fontsize=font_size)
            # else:
            #     axr[1,1].set_title(selected_filter +' G with Cut-off = ' + str(cutoff), fontsize=font_size)
            # axr[1,2].plot(x,filt[int(shape[0] / 2),int(shape[1]/ 2):shape[1]])
            # axr[1,0].set_xticklabels('')
            # axr[1,1].set_xticklabels('')
            # axr[1,0].set_yticklabels('')
            # axr[1,1].set_yticklabels('')

            # axr[2,0].imshow(filt_img, cmap='gray')
            # axr[2,0].set_title('Filtered image', fontsize=font_size)
            # axr[2,1].imshow(mag_filt_dft, cmap='gray')
            # axr[2,1].set_title('Filtered image F*G in Frequency Domain', fontsize=font_size)
            # axr[2,0].set_xticklabels('')
            # axr[2,1].set_xticklabels('')
            # axr[2,0].set_yticklabels('')
            # axr[2,1].set_yticklabels('')
            # axr[2,2].plot(x, mag_filt_dft[int(shape[0]/ 2),int(shape[1]/ 2):shape[1]])

            # st.pyplot(fig)
            fig = make_subplots(rows=3, cols=3,
                            shared_xaxes=True, 
                            vertical_spacing=0.05,
                            horizontal_spacing=0.00025,
                            subplot_titles=['Original image','Image in Freq, F','',
                                            '','Filter, G', '',
                                            'Filterd image', 'Filterd image in Freq, F*G', ''])
            fig.add_trace(fig_org.data[0], 1, 1)
            fig.add_trace(px.imshow(mag_img_dft, binary_string=True).data[0], 1, 2)
            trace=go.Scatter(x=np.linspace(0,n,n, endpoint=False),
                    y=np.squeeze(mag_img_dft[int(shape[0]/ 2),int(shape[1]/ 2):shape[1]]),
                    line=dict(width=2),
                    showlegend=False)
            fig.add_trace(trace, 1, 3)
            
            fig.add_trace(px.imshow(mag_filt, binary_string=True).data[0], 2, 2)
            trace=go.Scatter(x=np.linspace(0,n,n, endpoint=False),
                    y=np.squeeze(filt[int(shape[0]/ 2),int(shape[1]/ 2):shape[1]]),
                    line=dict(width=2),
                    showlegend=False)
            fig.add_trace(trace, 2, 3)
            
            fig.add_trace(fig_filt.data[0], 3, 1)
            fig.add_trace(px.imshow(mag_filt_dft, binary_string=True).data[0], 3, 2)
            trace=go.Scatter(x=np.linspace(0,n,n, endpoint=False),
                    y=np.squeeze(mag_filt_dft[int(shape[0]/ 2),int(shape[1]/ 2):shape[1]]),
                    line=dict(width=2),
                    showlegend=False)
            fig.add_trace(trace, 3, 3)
            
            
            
            fig.update_layout(coloraxis_showscale=False)
            # fig.update_xaxes(showticklabels=False)
            # fig.update_yaxes(showticklabels=False)
            fig.update_layout(width=900, height=800)
            st.plotly_chart(fig, use_container_width=False)
    
st.write("---")
st.caption("Anucha Chaichana") 
st.caption("anucha.cha@mahidol.edu")
