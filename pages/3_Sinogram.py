import streamlit as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# import pydicom as dicom
# import streamlit_nested_layout
# import nibabel as nib
from skimage.transform import radon
from scipy import ndimage
# import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly import colors
# import base64
from PIL import Image

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

# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)

# ---- HEADER SECTION ----
with st.container():
    st.title("Profile, Projection, and Sinogram")
    st.write("---")
# ---- LOAD IMAGE ----
BASE_DIR = Path(__file__).resolve().parent
PRJ_DIR = BASE_DIR / 'images/sino_proj/Shepp_Logan_Prj.npy'
SLICE_DIR = BASE_DIR / 'images/sino_proj/Shepp_Logan_copy.npy'
SINO_DIR = BASE_DIR / 'images/sino_proj/shepp_logan_sinogram.npy'
# imageNames = [f.name for f in IMAGE_DIR.iterdir() if f.suffix=='.nii']

# img_path = IMAGE_DIR / sample_image
st.sidebar.header("Sinogram")

# with st.expander("CHANGE PROJECTION DATA"):
#     sample_image = st.radio('Choose sample projection', imageNames, index=0)
#     img_path = IMAGE_DIR / sample_image


# def read_dcm(imp_path):
#     ds = dicom.dcmread(img_path)
#     img = ds.pixel_array.astype(float)
#     scaled_image = (np.maximum(img, 0) / img.max()) * 255.0
#     disp_img = np.uint8(scaled_image)
#     return disp_img


# @st.cache_resource
# def read_nii(img_path):
#     img = nib.load(img_path)
#     a = np.array(img.dataobj)
#     a = np.flip(a, axis=0)
#     scaled_image = (np.maximum(a, 0) / a.max()) * 255.0
#     a = np.uint8(scaled_image)
#     return a

# @st.cache_data
# def get_projection(object, theta):
#     sino_p = np.zeros([m, theta.shape[0], n])
#     for i in range(n):
#         sino_p[:, :, i] = radon(object[:, :, i], theta, preserve_range=True)
    
#     prj = np.swapaxes(sino_p, 0, 1)
#     prj = np.swapaxes(prj, 1, 2)

#     return prj


# @st.cache_data
def disp_prj(prj,theta):
    n_col = 6
    n_row = int(prj.shape[0]/n_col)
    fig, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
    axs = axs.flatten()
    i = 0
    for img, ax in zip(prj, axs):
        ax.imshow(img, cmap="gray")
        ax.set_title('Angle: ' + str(theta[i]), size=5)
        i+=1
        ax.axis('off')
    fig.tight_layout()
    st.pyplot(fig)


# @st.cache_resource
def init():
    prj = np.load(PRJ_DIR)
    tomo = np.load(SLICE_DIR)
    sinogram = np.load(SINO_DIR)
    return prj, tomo, sinogram

prj, tomo, sinogram = init()

# disp_img = read_nii(img_path)
n_ang, n_z, n_x = np.shape(prj)

arc = 360

# Slice_img = np.array(Image.open(IMAGE_DIR / 'circle_square.bmp'))

# if "angle" not in st.session_state:
#     st.session_state.angle = 0
#     st.session_state.slice = int(m/2)
st.header("Projection")
with st.form("Projection"):
    left_low_col, right_low_col = st.columns([1, 1], gap="large")

    with left_low_col:
        start_ang_prj = st.radio(
            "Start angle (ϴ):", [0, 45, 90, 180], index=0, key='start_ang_prj', horizontal=True)
        step_ang_prj = st.radio(
            "Step angle (ϴ):", [3, 6, 12], index=2, key='step_ang_prj', horizontal=True)
        ang_range = st.radio(
            "Angular range (ϴ):", [180, 360], index=1, key='ang_range', horizontal=True)
        rot_direct = st.radio(
            "Direction:", ["CW", "CCW"], index=1, key='rot_direction', horizontal=True)
        submitted = st.form_submit_button("Apply")
        if submitted:

            # theta = np.array(range(start_ang_prj, start_ang_prj+ang_range , step_ang_prj))
            if rot_direct == 'CW':
                theta = np.array(range(start_ang_prj, start_ang_prj+ang_range , step_ang_prj))
                ang_dir = 'clockwise'
                wrapped_range = [(start_ang_prj + i) % 360 for i in range(0, ang_range, step_ang_prj)]
            else:
                theta = np.array(range(start_ang_prj, start_ang_prj-ang_range , -step_ang_prj))
                ang_dir = 'counterclockwise'
                wrapped_range = [(start_ang_prj - i) % 360 for i in range(0, ang_range, step_ang_prj)]
            
            # ang_idx = range(0, 120, int(step_ang_prj/3))
            # ang_idx = list(ang_idx)
            # selected_index = ang_idx.index(start_ang_prj)  # Assuming 0 is the selected index
            # ang_idx = ang_idx[selected_index:] + ang_idx[:selected_index]
            # prj_ = prj[ang_idx,:,:]
            
            # start = 90
            # step = 3
            # ang_range = 360
            # wrapped_range = [(start_ang_prj + i) % 360 for i in range(0, ang_range, step_ang_prj)]
            ang_idx = range(357, -1, -3)
            ang_idx = list(ang_idx)
            # ang_idx = [(180 + i) % 360 for i in range(0, 360, 3)]
            indices = [ang_idx.index(value) for value in wrapped_range if value in ang_idx]
            prj_ = np.zeros((len(indices), np.shape(prj)[1], np.shape(prj)[2]), dtype=int)
            for i in range(len(indices)):
                prj_[i, :, :] = prj[indices[i], :, :]
                        
            r = np.ones(np.shape(theta))
            fig = go.Figure()
            trace=go.Scatterpolar(r=r,
                                theta=theta,
                                opacity=1,
                                mode = 'markers',)
            fig.add_trace(trace)
            fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            fig.update_layout(coloraxis_showscale=False)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            fig.add_layout_image(
                dict(
                    source=Image.open(BASE_DIR / 'images/sino_proj/shepp_logan.png'),
                    xref="x domain",
                    yref="y domain",
                    x=0.5,
                    y=0,
                    sizex=1,
                    sizey=1,
                    sizing="contain",
                    xanchor="center",
                    yanchor="bottom",
                    opacity=0.5,
                    layer="above")
            )
            fig.update_layout(
                template=None,
                polar = dict(
                    radialaxis = dict(range=[0, 1], showticklabels=False, ticks=''),
                    angularaxis = dict(rotation = 90,
                    direction = 'clockwise', tickfont_size=8)
            ))
            fig.update_layout(width=500, height=500)
            st.plotly_chart(fig, use_container_width=True, interactivity=False)

            with right_low_col:
                fig_sino = px.imshow(prj_, animation_frame=0, binary_string=True, labels=dict(animation_frame="Projection"))
                fig_sino.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
                fig_sino.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_sino, use_container_width=True)
            

# with st.containers('Profile'):
st.header("Profile")
with st.form("Profile"):
    left_top_col, right_top_col = st.columns([1,1], gap="large")
    with left_top_col:

        profile_ang = st.number_input("Angle (ϴ):", min_value=0, max_value=359,
                        step=6, value=0)
        
        submitted = st.form_submit_button("Apply")
        if submitted:

            with right_top_col:

                    Slice_img = tomo.copy()
                    Slice_img = ndimage.rotate(Slice_img, -profile_ang, reshape=False)
                    ang_idx = int(profile_ang*119/359)
                    profile = sinogram[ang_idx, :]

                    
                    fig_org = px.imshow(Slice_img, binary_string=True)
                    fig_org.update_layout(
                        autosize=False,
                        width=340,
                        height=500,
                        margin=dict(
                            l=5,
                            r=5,
                            b=5,
                            t=5,
                            pad=4
                        ),)
                    st.plotly_chart(fig_org, use_container_width=True)
                    
                    profile_img = profile[np.newaxis, ...]
                    fig_profle = px.imshow(profile_img, binary_string=True)

                    fig_profle.update_layout(
                        autosize=False,
                        width=340,
                        height=100,
                        margin=dict(
                            l=5,
                            r=5,
                            b=5,
                            t=5,
                            pad=4
                        ),)
                    st.plotly_chart(fig_profle, use_container_width=True)
                    
                    
                    fig = go.Figure()
                    trace=go.Scatter(x=np.linspace(0,n_x,n_x, endpoint=False),
                            y=profile,
                            line=dict(width=2),
                            showlegend=False)
                    fig.add_trace(trace)
                    fig.update_layout(
                        autosize=False,
                        width=340,
                        height=200,
                        margin=dict(
                            l=5,
                            r=5,
                            b=5,
                            t=5,
                            pad=4
                        ),)
                    st.plotly_chart(fig, use_container_width=True)
        
# with st.expander('Sinogram'):
st.header("Sinogram")
with st.form("Sinogram"):
    left_mid_col, right_mid_col = st.columns([1,1], gap="large")

    with left_mid_col:
        step_angle = st.radio(
            "Step angle (ϴ):", [1, 3, 6, 12], index=2, horizontal=True)
        submitted = st.form_submit_button("Apply")
        if submitted:
        # slice_sino = st.number_input(
        #     "Slice:", min_value=1, max_value=n_z, step=5, value=int(n_z/2), key='slice_sino')
        # start_angle = st.selectbox("Start angle (ϴ):", [0, 45, 90, 135, 180, 225, 270, 315],index=0)
        
        
        # fig = go.Figure()
        # fig.add_trace(fig_org.data[0], 1, 1)
        
        # Img_sino = disp_img[:, :, slice_sino-1].copy()
            theta = np.array(range(0, arc, step_angle))
            # theta_r = np.pi * theta/180

            # fig3 = plt.figure()
            # axes_coords = [0.1, 0.1, 0.8, 0.8]
            # ax4 = fig3.add_axes(axes_coords)
            # ax4.imshow(Img_sino, cmap="gray")
            # ax4.axis('off')

            # ax3 = fig3.add_axes(axes_coords, projection='polar')
            # ax3.patch.set_alpha(0)
            # ax3.set_theta_zero_location("N")
            # ax3.set_theta_direction(-1)
            r = np.ones(np.shape(theta))
            # ax3.scatter(theta_r, r,
            #          color="tab:orange", lw=1,)

            # ax3.set_rlabel_position(-22.5)
            
            # st.pyplot(fig3)
            # fig = make_subplots(rows=1, cols=1)
            fig = go.Figure()
            # fig_org = px.imshow(tomo, binary_string=True)
            # # fig = px.imshow(tomo, color_continuous_scale='gray')
            # fig.add_trace(fig_org.data[0])
            trace=go.Scatterpolar(r=r,
                                theta=theta,
                                opacity=1,
                                mode = 'markers',)
            fig.add_trace(trace)
            fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            fig.update_layout(coloraxis_showscale=False)
            # fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            # img = base64.b64encode(open(BASE_DIR / 'images/sino_proj/shepp_logan.png', 'rb').read())
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            fig.add_layout_image(
                dict(
                    source=Image.open(BASE_DIR / 'images/sino_proj/shepp_logan.png'),
                    xref="x domain",
                    yref="y domain",
                    x=0.5,
                    y=0,
                    sizex=1,
                    sizey=1,
                    sizing="contain",
                    xanchor="center",
                    yanchor="bottom",
                    opacity=0.5,
                    layer="above")
            )
            fig.update_layout(
                template=None,
                polar = dict(
                radialaxis = dict(range=[0, 1], showticklabels=False, ticks=''),
                angularaxis = dict(rotation = 90,
                    direction = "clockwise", tickfont_size=8)
                ))
            fig.update_layout(width=500, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # fig.add_trace(fig_org.data[0], 1, 1)
        
            with right_mid_col:
                # Slice_img = disp_img[:, :, slice_sino-1].copy()
                # Slice_img = tomo.copy()
                # sinogram = np.zeros([360,n])
                # for i in theta:
                #     sinogram[i,:] = radon(Slice_img, [i], preserve_range=True)[:,0]
                
                
                # fig4, ax5 = plt.subplots()
                # fig4.set_figwidth(8)

                # ax5.imshow(sinogram, cmap="gray")
                # ax5.invert_yaxis()
                # plt.tight_layout()
                # st.pyplot(fig4)
                
                ang_idx = range(0, 359, step_angle)
                # ang_idx = theta*119/359
                # ang_idx = ang_idx.astype(int).item()
                disp_sino = np.zeros(np.shape(sinogram))
                disp_sino[ang_idx, :] = sinogram[ang_idx, :]
                fig_sino = px.imshow(disp_sino, color_continuous_scale='gray')
                fig_sino.update_layout(width=500, height=1000)
                fig_sino.update_layout(coloraxis_showscale=False)
                # fig_sino.update_xaxes(showticklabels=False)
                # fig_sino.update_yaxes(showticklabels=False)
                st.plotly_chart(fig_sino, use_container_width=False)
                
                

st.write("---")
st.caption("Anucha Chaichana")
st.caption("anucha.cha@mahidol.edu")
