import streamlit as st
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import streamlit_nested_layout
import nibabel as nib
from scipy import ndimage

st.set_page_config(page_title="Sinogram", page_icon="✋🏻", layout="wide")

remove_top_padding = """
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
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

uploaded_file = st.file_uploader("Upload MUGA file", type=['dcm','DCM'])
if uploaded_file is not None:
    # img_path = uploaded_file
    # st.write("filename:", uploaded_file.name)
    ds = dicom.dcmread(uploaded_file)
    img = ds.pixel_array.astype(float)
    scaled_image = (np.maximum(img, 0) / img.max()) * 255.0
    img = np.uint8(scaled_image[0])
    
    # img = nib.load(uploaded_file)
    # a = np.array(img.dataobj)
    # a = np.flip(a, axis=0)
    # scaled_image = (np.maximum(a, 0) / a.max()) * 255.0
    # img = np.uint8(scaled_image[0])

    org_img = Image.fromarray(img)
    st.image(org_img)