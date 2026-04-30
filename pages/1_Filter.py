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
    return img
        
def pad_image(img):
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

# ---- STORE IMAGES IN STATE
if 'compare_filter' not in st.session_state or not isinstance(st.session_state.compare_filter, list):
    st.session_state.compare_filter = []
if 'last_filt' not in st.session_state:
    st.session_state.last_filt = None
if 'last_filt_str' not in st.session_state:
    st.session_state.last_filt_str = ""

# with st.sidebar:
# with st.expander("CHANGE INPUT IMAGE"):
left_top_col, right_top_col  = st.columns((1,1))
with left_top_col:
    
    sample_image = st.radio('Choose sample image', imageNames, index=0)
    
with right_top_col:

    uploaded_file = st.file_uploader("or Upload image", accept_multiple_files=False, type=['dcm','DCM','jpg','JPG','jpeg','png'])
    if uploaded_file is not None:
        img_path = uploaded_file
        img = read_image(img_path)
        # st.write("filename:", uploaded_file.name)
    else:
        img_path = os.path.join(IMAGE_DIR, sample_image)
        img = np.array(Image.open(img_path).convert('L'))
        # img_path = IMAGE_DIR / sample_image

st.write("---")

img = pad_image(img)
m,n = np.shape(img)
shape = np.shape(img)
x = np.linspace(0,1,int(m/2))
filt_img = None

# ---- FILTER ----
Filter_list = ['Gaussian', 
        'Butterworth Low-pass', 
        'Butterworth High-pass',
        'Hanning']

left_col, right_col  = st.columns(2, gap="large")

with left_col:
    st.subheader('Filter')
    selected_filter = st.radio("Filter type:", Filter_list)

    if selected_filter == Filter_list[0]:
        fwhm = st.slider("Full-width at half maximum (pixel):", min_value=1.0, max_value=21.0, step=2.0, value=3.0)
        sigma = fwhm/(2.355)
        radius = int(4 * sigma + 0.5)
        ksize = 2 * radius + 1
        filt_img = gaussian(img, sigma, truncate=4, preserve_range=True, mode='reflect')
        filt_str = f"{selected_filter} FWHM={fwhm}"
        
        gauss_ker = gaussianKernel2(ksize, sigma,  twoDimensional=False)
        df = pd.DataFrame({'x':range(len(gauss_ker)), 'y':gauss_ker})
        line_chart = alt.Chart(df).mark_line(interpolate='basis').encode(
            alt.X('x', title='Pixels'),
            alt.Y('y', title='Kernel weight')
        )
        st.altair_chart(line_chart, use_container_width=True)

    elif 'Butterworth' in selected_filter:
        cutoff = st.slider("Cut-off:", min_value=0.05, max_value=1.0, step=0.05, value=0.5)
        order = st.slider("Order:", min_value=1, max_value=10, step=1, value=3)
        if 'Low-pass' in selected_filter:
            filt = getButterworth_lowpass_filter(shape, cutoff, order)
        else:
            filt = getButterworth_highpass_filter(shape, cutoff, order)
        filt_img = fourier_filter(img, filt)
        filt_str = f"{selected_filter} c={cutoff}, n={order}"
        
        df = pd.DataFrame({'x':x, 'y':filt[int(shape[0]/ 2), int(shape[1]/ 2):shape[1]]})
        line_chart = alt.Chart(df).mark_line(interpolate='basis').encode(
            alt.X('x', title='Frequency (cycle/pixel)'),
            alt.Y('y', title='Amplitude')
        )
        st.altair_chart(line_chart, use_container_width=True)
                
    else: # Hanning
        cutoff = st.slider("Cut-off:", min_value=0.05, max_value=1.0, step=0.05, value=0.5)
        filt = getHanning_filter(shape, cutoff)
        filt_img = fourier_filter(img, filt)
        filt_str = f"{selected_filter} c={cutoff}"
        
        df = pd.DataFrame({'x':x, 'y':filt[int(shape[0]/ 2), int(shape[1]/ 2):shape[1]]})
        line_chart = alt.Chart(df).mark_line(interpolate='basis').encode(
            alt.X('x', title='Frequency (cycle/pixel)'),
            alt.Y('y', title='Amplitude')
        )
        st.altair_chart(line_chart, use_container_width=True)

    if filt_img is not None:
        st.session_state.last_filt = filt_img[0:m, 0:n]
        st.session_state.last_filt_str = filt_str

with right_col:
    fig_org = px.imshow(img, binary_string=True)
    fig_org.update_xaxes(showticklabels=False)
    fig_org.update_yaxes(showticklabels=False)
    fig_org.update_layout(coloraxis_showscale=False)
    fig_org.update_layout(width=340, title_text="Original Image")
    st.plotly_chart(fig_org, use_container_width=False)
   
    
    if st.session_state.last_filt is not None:
        fig_filt = px.imshow(st.session_state.last_filt, binary_string=True)
        fig_filt.update_xaxes(showticklabels=False)
        fig_filt.update_yaxes(showticklabels=False)
        fig_filt.update_layout(width=340, title_text="Filtered Image")
        st.plotly_chart(fig_filt, use_container_width=False)
        st.write(st.session_state.last_filt_str)
        
        if st.button("Add to Comparison"):
            st.session_state.compare_filter.append({
                'name': f"{sample_image} - {st.session_state.last_filt_str}",
                'data': st.session_state.last_filt
            })
            st.success(f"Added to comparison list ({len(st.session_state.compare_filter)} images total)")
        
st.write("---")

# --- COMPARISON GALLERY ---
if len(st.session_state.compare_filter) > 0:
    st.subheader("Comparison Gallery")
    if st.button("Clear Comparison"):
        st.session_state.compare_filter = []
        st.rerun()

    # Always include Original in comparison if desired, or just the saved ones
    all_to_compare = [{'name': 'Original', 'data': img}] + st.session_state.compare_filter
    num_images = len(all_to_compare)
    
    titles = [item['name'] for item in all_to_compare]
    cols_per_row = 3
    num_rows = (num_images + cols_per_row - 1) // cols_per_row
    
    fig_comp = make_subplots(rows=num_rows, cols=min(num_images, cols_per_row), subplot_titles=titles)
    
    for i, img_obj in enumerate(all_to_compare):
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

st.write("---")

display = st.checkbox('More details')

if display and (st.session_state.last_filt is not None):
    filt_img = st.session_state.last_filt
    cols = colors.DEFAULT_PLOTLY_COLORS
    if selected_filter == Filter_list[0]:
        # sigma = fwhm/(2.355)
        # radius = int(4 * sigma + 0.5)
        # ksize = 2 * radius + 1
        # gauss_2D = gaussianKernel2(ksize, sigma)
        
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
        fig.add_trace(fig_filt.data[0], 1, 3)
        trace=go.Scatter(x=np.linspace(0,n,n, endpoint=False),
                 y=np.squeeze(filt_img[int(shape[0]/ 2),:]),
                 line=dict(width=2),
                 showlegend=False)
        fig.add_trace(trace, 3, 1)
        
        # Need to recompute these for the detail view
        sigma_detail = float(st.session_state.last_filt_str.split('=')[-1]) if 'FWHM' in st.session_state.last_filt_str else 1.0
        sigma_detail = sigma_detail/2.355
        ksize_detail = int(4 * sigma_detail + 0.5) * 2 + 1
        gauss_2D = gaussianKernel2(ksize_detail, sigma_detail)

        trace=go.Scatter(x=np.linspace(0,ksize_detail,ksize_detail, endpoint=False),
            y=np.squeeze(gauss_2D[int(ksize_detail/2),:]),
            line=dict(width=2),
            showlegend=False)
        fig.add_trace(trace, 1, 5)
        
        g_kernel = px.imshow(gauss_2D, binary_string=True)
        g_kernel.update_xaxes(showticklabels=False)
        g_kernel.update_yaxes(showticklabels=False)
        fig.add_trace(g_kernel.data[0], 2, 5)
        
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(width=700, height=800)
        st.plotly_chart(fig, use_container_width=False)
            

    else:
            intro_markdown = read_markdown_file(os.path.join(BASE_DIR, "markdown/Text directory.md"))
            st.markdown(intro_markdown, unsafe_allow_html=True)

            image_fft = np.fft.fft2(img)
            shift_fft = np.fft.fftshift(image_fft)
            mag_img_dft = np.log(np.abs(shift_fft)+1)
            
            # Re-generate the filter used for the last result
            if 'Butterworth' in st.session_state.last_filt_str:
                parts = st.session_state.last_filt_str.split(',')
                c = float(parts[0].split('=')[-1])
                o = int(parts[1].split('=')[-1])
                filt_current = getButterworth_lowpass_filter(shape, c, o)
            else:
                c = float(st.session_state.last_filt_str.split('=')[-1])
                filt_current = getHanning_filter(shape, c)

            filt_dft = np.multiply(filt_current, shift_fft)
            mag_filt_dft = np.log(np.abs(filt_dft)+1)
            mag_filt= np.log(np.abs(filt_current)+1)

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
                    y=np.squeeze(filt_current[int(shape[0]/ 2),int(shape[1]/ 2):shape[1]]),
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
            fig.update_layout(width=900, height=800)
            st.plotly_chart(fig, use_container_width=False)
    
st.write("---")
st.caption("Anucha Chaichana") 
st.caption("anucha.cha@mahidol.ac.th")
