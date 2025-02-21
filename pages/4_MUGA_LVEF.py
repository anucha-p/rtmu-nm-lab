import math
import pandas as pd
import scipy
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
# from PIL import Image
from pydicom import dcmread
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
# import streamlit_nested_layout
# import nibabel as nib
# from scipy import ndimage
import torch
from src.unet3d.model import UNet512, UNet                 
import re
# import os
import altair as alt
st.set_page_config(page_title="Auto MUGA EF Analysis", page_icon="✋🏻", layout="wide")

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

@st.cache_resource
def load_model(model_path):
    with torch.no_grad():
        model = UNet(img_ch=1, output_ch=1)
        state_dict = torch.load(model_path, weights_only=True, map_location = device)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

BASE_DIR = Path(__file__).resolve().parent
# BASE_DIR = os.path.abspath(os.path.join(__file__, '../'))
# IMAGE_DIR = os.path.join(BASE_DIR, 'images/filter')
model_path = BASE_DIR / "Best_model_unet1024.pth"
model = load_model(model_path)
result = None
dcm_img = None

def read_dcm(dcm_img):
    ds = dcmread(dcm_img)
    arr = ds.pixel_array
    return arr

def normalize(x):
    norm_x = (x-np.min(x))/(np.max(x)-np.min(x))
    return norm_x


def display_polar(ds):
    arr = ds.pixel_array
    arr = normalize(arr)*255.0
    # arr = apply_color_lut(arr, palette='PET')
    arr = np.uint8(arr)
    st.image(arr, use_column_width=True)


def check_id(ds, id_input):
    ds_id = str(ds.PatientID)
    if ds_id == id_input:
        pt_id = ':green[Patient ID: ' + ds_id + ']'
    else:
        pt_id = ':red[Patient ID: ' + ds_id + ']'
    st.write(pt_id)

def text_field(label, columns=(1,2), **input_params):
    c1, c2 = st.columns(columns)

    # Display field name with some alignment
    # c1.markdown("##")
    original_title = f'<p style="font-size: 16px;">{label}</p>'
    c1.markdown(original_title, unsafe_allow_html=True)
    # c1.markdown(f'''
    # **:red[{label}]**''')

    # Sets a default key parameter to avoid duplicate key errors
    input_params.setdefault("key", label)

    # Forward text input parameters
    return c2.text_input(label, label_visibility='collapsed',**input_params)

def replace_zero_with_average(arr):
    # Copy the array to avoid modifying the original while iterating
    result = arr.copy()
    # Loop from index 1 to len(arr)-2 to avoid boundary issues
    for i in range(1, len(result)-1):
        if result[i] == 0:
            # Replace the 0 with the average of the previous and next element
            result[i] = (result[i-1] + result[i+1]) / 2
    return result

def find_ED_ES_idx(muga, mask_np):
    muga_masked = np.multiply(muga, mask_np)
    if np.sum(muga_masked) == 0:
        ED_idx = 0
        ES_idx = 0
    else:
        lv_area = np.sum(mask_np, axis = (1,2))
        lv_count = np.sum(muga_masked, axis = (1,2))
        lv_count = replace_zero_with_average(lv_count)
    #   lv_area[lv_area == 0] = np.nan
        # st.write(lv_area)
        lv_area = replace_zero_with_average(lv_area)
        # st.write(lv_area)
        bkg_idx = np.nanargmax(lv_count)
        st.write(bkg_idx)
        # ES_idx = np.nanargmin(sum_count)
        mask_ed = mask_np[bkg_idx, :, :]
        bkg_roi = auto_bkg_roi(mask_ed)
        bkg = muga*bkg_roi[np.newaxis,:,:]
        bkg_count = np.sum(bkg, axis = (1,2))
        
        Norm_count = lv_count - (bkg_count*lv_area/np.sum(bkg_roi))
        
        avg_bkg_count = np.sum(np.multiply(muga[bkg_idx,:,:], bkg_roi)) /np.sum(bkg_roi)
        
        
        st.write(Norm_count)
        st.write(avg_bkg_count)

        ED_idx = np.nanargmax(Norm_count)
        st.write(ED_idx)
        ES_idx = np.nanargmin(Norm_count)
        st.write(ES_idx)
        
        EF = (Norm_count[ED_idx] - Norm_count[ES_idx])*100/Norm_count[ED_idx]
        st.write(EF)
        
    return ED_idx, ES_idx, Norm_count, EF, bkg_roi

def dilate(mask,n):
  mask_dilate = np.copy(mask)
  for i in range(n):
    mask_dilate = scipy.ndimage.binary_dilation(mask_dilate).astype(mask_dilate.dtype)
  return mask_dilate

def mask_out_zeros(mask):
  mask_nan = np.copy(mask)
  mask_nan[mask_nan<1] = np.nan
  return mask_nan

def lvef(ED_image, ES_image, ED_roi, ES_roi):
    ED_count = np.sum(ED_roi*ED_image)
    st.write(ED_count)
    ES_count = np.sum(ES_roi*ES_image)
    st.write(ES_count)
    ED_bkg_roi = auto_bkg_roi(ED_roi)
    ES_bkg_roi = auto_bkg_roi(ES_roi)
    Norm_ED_count = ED_count - (np.sum(ED_bkg_roi*ED_image)*np.sum(ED_roi)/np.sum(ED_bkg_roi))
    Norm_ES_count = ES_count - (np.sum(ES_bkg_roi*ES_image)*np.sum(ES_roi)/np.sum(ES_bkg_roi))
    st.write(Norm_ED_count)
    st.write(Norm_ES_count)
    st.write(np.sum(ED_bkg_roi*ED_image)/np.sum(ED_bkg_roi))
    average_bkg = np.sum(ED_bkg_roi*ED_image)/np.sum(ED_bkg_roi)
    avg_ED_count = ED_count/np.sum(ED_roi)
    avg_ES_count = ES_count/np.sum(ES_roi)
    Norm_ED_count = (avg_ED_count - average_bkg)*np.sum(ED_roi)
    Norm_ES_count = (avg_ES_count - average_bkg)*np.sum(ES_roi)
    LVEF = (Norm_ED_count - Norm_ES_count)*100/Norm_ED_count
    return LVEF, Norm_ED_count, Norm_ES_count
  
def auto_bkg_roi(ROI):
    # find center of ROI
    cy, cx = scipy.ndimage.center_of_mass(ROI)
    cx = math.ceil(cx)
    cy = math.ceil(cy)
    
    # create background ROI
    LV_roi = np.copy(ROI)
    crop = np.zeros((np.shape(LV_roi)))
    crop[cy+2:,cx+2:] = 1
    # plt.imshow(crop)
    LV_dilate_end = dilate(LV_roi,5)
    LV_dilate_start = dilate(LV_roi,2)
    BKG_roi = crop * (LV_dilate_end - LV_dilate_start)

    return BKG_roi

def predict_img(model, img):
    img = img[np.newaxis, np.newaxis, ...]
    img = torch.tensor(img)
    img = img / img.max()
    img = img.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(img).cpu()
        # output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')

        mask = torch.sigmoid(output) > 0.5
    return mask[0].long().squeeze().numpy()
    
    
# st.subheader("**:blue[Upload MUGA file]**")
uploaded_file = st.file_uploader("Upload MUGA file", type=['dcm','DCM'], label_visibility="collapsed")
if uploaded_file is not None:
    # img_path = uploaded_file
    # st.write("filename:", uploaded_file.name)
    ds = dicom.dcmread(uploaded_file)
    dcm_img = ds.pixel_array.astype(float)
    
    scaled_image = (np.maximum(dcm_img, 0) / dcm_img.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    # st.write(dcm_img.shape)
    # org_img = Image.fromarray(scaled_image[0])
    # st.image(org_img)
    
    mask = np.zeros(dcm_img.shape)
    for i in range(dcm_img.shape[0]):
        img = dcm_img[i,:,:]
        mask[i] = predict_img(model,img)
    
    # st.image(mask[0])
    

if dcm_img is not None:
    # st.write(ds)

    num_frame = ds.get((0x0028, 0x0008))
    
    frame_time = ds['GatedInformationSequence'][0]['DataInformationSequence'][0]['FrameTime'].value
    beat_accept = ds['GatedInformationSequence'][0]['DataInformationSequence'][0]['IntervalsAcquired'].value
    beat_reject = ds['GatedInformationSequence'][0]['DataInformationSequence'][0]['IntervalsRejected'].value
    heart_rate = ds['HeartRate'].value
    st.write(num_frame.value)
  
    df_acquire = pd.DataFrame(
                {
                    "Para": ["Heart Rate", "Time/Frame", "Beats Accepted", "Beats Rejected"],
                    "Value": [heart_rate, frame_time, beat_accept, beat_reject],
                }
            )

    with st.container():
        # st.subheader("**:blue[Study Information]**")
        col1, col2, col3 = st.columns((1, 1, 1), gap="medium")
        with col1:
            name = text_field("Patient Name:", value=ds.PatientName)
            # patient_height = ds.get((0x0010, 0x1020))
            # if patient_height is not None and patient_height.value > 0.0:
            #     height = st.number_input("Height (m):", value=patient_height.value)
            # else:
            #     height = st.number_input("Height (m):", value=None, placeholder="Height in Meter")
            st.dataframe(df_acquire, hide_index=True)
        with col2:
            ptid = text_field("Patient ID:", value=ds.PatientID)
            patient_weight = ds.get((0x0010, 0x1030))
            if patient_weight is not None and patient_weight.value > 0.0:
                        # text_field('Weight(kg):', value=patient_weight.value)
                weight = st.number_input("Weight (kg):", value=patient_weight.value)
            else:
                weight = st.number_input("Weight (kg):", value=None, placeholder="Weight in kilogram")

            
        with col3:
            patient_age = ds.get((0x0010, 0x1010))
            if patient_age is not None:
                age = patient_age.value
                age = age.replace("Y", "")
                age = st.number_input("Age:", value=float(age))
            else:
                age = st.number_input("Age:", value=None, placeholder="Years")
            
            patient_sex = ds.get((0x0010, 0x0040))
            if patient_sex is not None:
                        # text_field('Gender:', value=patient_sex.value)
                if patient_sex.value == 'F':
                    index = 0
                else:
                    index = 1
                gender = st.radio("Gender:", ["Female", "Male"], index = index)
    with st.container():
        st.subheader("**:blue[Analysis Result]**")      
        pred_ed_idx, pred_es_idx, Norm_count, EF, Bkg_roi = find_ED_ES_idx(dcm_img,mask)

        pred_ED_image = dcm_img[pred_ed_idx,:,:]
        pred_ES_image = dcm_img[pred_es_idx,:,:]
        pred_ED_roi = mask[pred_ed_idx,:,:]
        pred_ES_roi = mask[pred_es_idx,:,:]
        
        if pred_ed_idx == pred_es_idx:
            EF_unet = 0
            pred_ED_count = 0
            pred_ES_count = 0
        else:
            BKG_ED_roi = auto_bkg_roi(pred_ED_roi)
            BKG_ES_roi = auto_bkg_roi(pred_ES_roi)
            EF_unet, pred_ED_count, pred_ES_count = lvef(pred_ED_image, pred_ES_image, pred_ED_roi, pred_ES_roi)
        
        # st.write(EF_unet)
        # ed_frame = text_field('ED frame:', value=pred_ed_idx)
       
        col21, col22 = st.columns((2, 3), gap="medium")
        with col21:
            df = pd.DataFrame({
                    'Frame':range(1,25,1),
                    'Count':np.round(Norm_count)})
            # line_chart = alt.Chart(df).mark_line(interpolate='basis').encode(
            #         alt.X('x', title='Frame'),
            #         alt.Y('y', title='Count'),
            #     )
            line_chart = alt.Chart(df).mark_line(interpolate='natural', point=True).encode(
                    x='Frame',
                    y='Count',
                    tooltip=['Frame', 'Count'],
                )
            st.altair_chart(line_chart, use_container_width=True)
            # st.line_chart(Norm_count, x="Frame", y="Count")
            
            with st.expander("Review MUGA Image and ROI"):
                frame = st.slider("Select Frame", min_value=1, max_value=24, value=1, step=1)
                # org_img = Image.fromarray(scaled_image[frame-1])
                col211, col212 = st.columns([1, 1], gap='large')
                col211.image(scaled_image[frame-1], use_container_width=True)
                col212.image(mask[frame-1], use_container_width=True)
            # st.number_input("ED frame:", min_value=0, max_value=24, value=pred_ed_idx)
            # st.number_input("ES frame:", min_value=0, max_value=24,value=pred_es_idx)
            # st.number_input("Ejection Fraction:", min_value=0.0, max_value=100.0, value=EF_unet, disabled=True)
            
        with col22:
            fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))
            # axs[0].set_title("Count")
            # axs[0].plot(Norm_count)
            
            axs[0].set_axis_off()
            ed_frame_idx = pred_ed_idx+1
            axs[0].set_title("End Diastoric Frame: %d" %ed_frame_idx)
            axs[0].imshow(pred_ED_image, cmap='CMRmap')
            axs[0].imshow(mask_out_zeros(pred_ED_roi), cmap='Greys_r', alpha=0.5)
            axs[0].imshow(mask_out_zeros(Bkg_roi), cmap='Greys', alpha=0.5)
            
            axs[1].set_axis_off()
            es_frame_idx = pred_es_idx+1
            axs[1].set_title("End Systoric Frame: %d" %es_frame_idx)
            axs[1].imshow(pred_ES_image, cmap='CMRmap')
            axs[1].imshow(mask_out_zeros(pred_ES_roi), cmap='Greys_r', alpha=0.5)
            axs[1].imshow(mask_out_zeros(Bkg_roi), cmap='Greys', alpha=0.5)
            st.pyplot(fig)
            
            col221, col222 = st.columns([1, 1])
            col221.metric(label="Ejection Fraction (%)", value="%.2f" %EF, border=True)
            df = pd.DataFrame(
                {
                    "Phase": ["End Diastoric", "End Systoric"],
                    "Frame Index": [ed_frame_idx, es_frame_idx],
                    "Norm Count": [np.round(Norm_count[pred_ed_idx]), np.round(Norm_count[pred_es_idx])],
                }
            )
            col222.dataframe(df, hide_index=True)