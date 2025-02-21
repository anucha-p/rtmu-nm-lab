# import asyncio

from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid 
from streamlit_extras.add_vertical_space import add_vertical_space

from PIL import Image
import streamlit as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# import pydicom as dicom
# import streamlit_nested_layout
import nibabel as nib
# from skimage.transform import radon
# from scipy import ndimage
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# from plotly import colors
import base64
import SimpleITK as sitk

# import pyvista as pv
# from stpyvista import stpyvista
# from pyvista import examples
import streamlit.components.v1 as components
# import vtk
# from streamlit_drawable_canvas import st_canvas
# from ipywidgets import embed
# from itkwidgets import view
# from vtkmodules.vtkInteractionImage import vtkImageViewer2

# loop = asyncio.new_event_loop()
# asyncio.set_event_loop(loop)

st.set_page_config(page_title="Sinogram", page_icon="✋🏻", layout="wide")

remove_top_padding = """
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 2rem;
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

parameter_label = """{
                        border: 1px solid rgba(49, 51, 63, 0.2);
                        border-radius: 0.5rem;
                        padding: calc(1em - 1px);
                        background-color: #DCDCDC;
                    }
                    """
                    
border = """{
                        border: 1px solid rgba(49, 51, 63, 0.2);
                        border-radius: 0.5rem;
                        padding: calc(1em - 1px);
                    }
                    """  
                    
parameter_button = """
                    button {
                        background-color: #DCDCDC;
                        color: black;
                        border-radius: 10px;
                    }
                    """
parameter_selectbox = """
                    selectbox {
                        background-color: DarkSlateGray;
                        color: #4682B4;
                        border-radius: 10px;
                    }
                    """

st.markdown(remove_top_padding, unsafe_allow_html=True)

# ---- Function ----
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

def selectbox(label, columns=(1,1), options=[], **input_params):
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
    return c2.selectbox(label, options=options, index=1, label_visibility='collapsed',**input_params)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded_img = base64.b64encode(img_bytes).decode()
    return encoded_img

def load_and_store_dicom_series(directory, session_key):
    if session_key not in st.session_state:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory)
        reader.SetFileNames(dicom_names)
        image_sitk = reader.Execute()
        image_np = sitk.GetArrayFromImage(image_sitk)
        st.session_state[session_key] = image_np
    return st.session_state[session_key]

def plot_slice(slice, size=(5, 5), is_nifti=False):
    # Adjust the figure size for consistent viewer sizes
    fig, ax = plt.subplots(figsize=size)
    # Calculate the square canvas size
    canvas_size = max(slice.shape)
    canvas = np.full((canvas_size, canvas_size), fill_value=slice.min(), dtype=slice.dtype)
    # Center the image within the canvas
    x_offset = (canvas_size - slice.shape[0]) // 2
    y_offset = (canvas_size - slice.shape[1]) // 2
    canvas[x_offset:x_offset+slice.shape[0], y_offset:y_offset+slice.shape[1]] = slice
    fig.patch.set_facecolor('black')  # Set the figure background to black
    ax.set_facecolor('black')
    if is_nifti:
        canvas = np.rot90(canvas)
    # else:
    #     canvas = canvas[::-1, ::-1]

    ax.imshow(canvas, cmap='gray')
    ax.axis('off')
    return fig

def get_scout(vol, session_key):
    if session_key not in st.session_state:
        scout = np.mean(vol, axis=1)

        # Normalize to range 0 to 1
        normalized_arr = (scout - np.min(scout)) / (np.max(scout) - np.min(scout))
        # Scale to range 0 to 255
        rescaled_arr = (normalized_arr * 255).astype(np.uint8)
        
        # image_pil = Image.fromarray(rescaled_arr)
        st.session_state[session_key] = rescaled_arr
    return st.session_state[session_key]
        

def load_nifti_file(filepath, session_key):
    if session_key not in st.session_state:
        nifti_img = nib.load(filepath)
        image_np = np.asanyarray(nifti_img.dataobj)
        st.session_state[session_key] = image_np
    return st.session_state[session_key]



# ---- HEADER SECTION ----
with st.container():
    st.title("CT Simulator")
    st.write("---")
# ---- LOAD IMAGE ----
BASE_DIR = Path(__file__).resolve().parent

with st.container():
    # Create three columns

    col1, col2, col3, col4 = st.columns([1,1.5,1,1.5], gap="large")

    # Add content to the first column

    with col1:

        st.subheader("Admin Panel")

        # st.write("This is the content of column 1.")
        col11, col12 = st.columns(2)
        with col11:
            
            image_base64 = img_to_bytes(BASE_DIR / 'images/CTsimulator/desktop.png')
            link = 'https://www.google.com/'
            html = f"<a href='{link}'><img src='data:image/png;base64,{image_base64}' style='width:100%; height:100%'></a>"
            st.markdown(html, unsafe_allow_html=True)

            
            image_base64 = img_to_bytes(BASE_DIR / 'images/CTsimulator/diskette.png')
            link = 'https://www.google.com/'
            html = f"<a href='{link}'><img src='data:image/png;base64,{image_base64}' style='width:100%; height:100%'></a>"
            st.markdown(html, unsafe_allow_html=True)
            
            image_base64 = img_to_bytes(BASE_DIR / 'images/CTsimulator/warning.png')
            link = 'https://www.google.com/'
            html = f"<a href='{link}'><img src='data:image/png;base64,{image_base64}' style='width:100%; height:100%'></a>"
            st.markdown(html, unsafe_allow_html=True)
            
        with col12:
            # clipboard_image = Image.open(BASE_DIR / 'images/CTsimulator/clipboard.png')
            # clipboard_image = np.array(clipboard_image)
            # st.image(clipboard_image)
            
            image_base64 = img_to_bytes(BASE_DIR / 'images/CTsimulator/clipboard.png')
            link = 'https://www.google.com/'
            html = f"<a href='{link}'><img src='data:image/png;base64,{image_base64}' style='width:100%; height:100%'></a>"
            st.markdown(html, unsafe_allow_html=True)
            
            # cogwheel_image = Image.open(BASE_DIR / 'images/CTsimulator/cogwheel.png')
            # cogwheel_image = np.array(cogwheel_image)
            # st.image(cogwheel_image)
            
            image_base64 = img_to_bytes(BASE_DIR / 'images/CTsimulator/cogwheel.png')
            link = 'https://www.google.com/'
            html = f"<a href='{link}'><img src='data:image/png;base64,{image_base64}' style='width:100%; height:100%'></a>"
            st.markdown(html, unsafe_allow_html=True)
            
            image_base64 = img_to_bytes(BASE_DIR / 'images/CTsimulator/folders.png')
            link = 'https://www.google.com/'
            html = f"<a href='{link}'><img src='data:image/png;base64,{image_base64}' style='width:100%; height:100%'></a>"
            st.markdown(html, unsafe_allow_html=True)
        

    with col2:

        st.subheader("Patient Information")

        pt_id = text_field('Patient ID', placeholder='0123456')
        pt_name = text_field('Patient Name', placeholder='John Doe')
        pt_position = selectbox('Patient Position',  options=['Supine', 'Prone'])
        pt_body = selectbox('Patient Body',  options=['Head First', 'Feet First'])
        
        if pt_position == 'Supine':
            pt_position_image = Image.open(BASE_DIR / 'images/CTsimulator/lying-down.png')
        else:
            pt_position_image = Image.open(BASE_DIR / 'images/CTsimulator/facing-down.png')
        pt_position_array = np.array(pt_position_image)
        
        if pt_body == 'Feet First':
            pt_position_array = np.fliplr(pt_position_array)
        col1, col2 = st.columns([2,3], gap="small")
        with col2:
            st.image(pt_position_array[90:420,:])
        
        series_desc = text_field('Series Description', placeholder='CT brain')

    with col3:

        st.subheader("Patient Protocol")

        # st.write("This is the content of column 3.")
        anatomy_image = Image.open(BASE_DIR / 'images/CTsimulator/anatomy.png')
        anatomy_image = np.array(anatomy_image)
        st.image(anatomy_image)
            
        
    with col4:

        st.subheader("Autoview")

        dicom_dir = '/home/anucha/python_projects/streamlit_app/pages/images/CTsimulator/CT1/'
        image_np = load_and_store_dicom_series(dicom_dir, "dicom_image_data")
        # axial_slice_num = st.slider(' ', 0, image_np.shape[0] - 1, 0, key="axial_slider")
        # fig = plot_slice(image_np[axial_slice_num, :, :], size=(3, 3), is_nifti=False)
        # st.pyplot(fig, clear_figure=True)
        
        scout = get_scout(image_np, 'scout')
        fig, ax = plt.subplots(figsize=(3,3))
        fig.patch.set_facecolor('black')  # Set the figure background to black
        ax.set_facecolor('black')
        scout = scout[::-1, ::-1]
        ax.imshow(scout, cmap='gray')
        ax.axis('off')
        st.pyplot(fig, clear_figure=True)
        # st.image(scout)
 
        # ###
        # ## Initialize a plotter object
        # plotter = pv.Plotter(window_size=[400, 400])

        # ## Create a mesh
        # mesh = pv.Sphere(radius=1.0, center=(0, 0, 0))

        # ## Associate a scalar field to the mesh
        # x, y, z = mesh.cell_centers().points.T
        # mesh["My scalar"] = z

        # ## Add mesh to the plotter
        # plotter.add_mesh(
        #     mesh,
        #     scalars="My scalar",
        #     cmap="prism",
        #     show_edges=True,
        #     edge_color="#001100",
        #     ambient=0.2,
        # )

        # ## Some final touches
        # plotter.background_color = "white"
        # plotter.view_isometric()

        # ## Pass a plotter to stpyvista
        # stpyvista(plotter)
        
        # ###
        
        # path = dicom_dir
        # reader = pv.DICOMReader(path)
        # dataset = reader.read()
        
        # plotter2 = pv.Plotter(window_size=[400, 400])
        # mesh2 = dataset
        
        # vol = plotter2.add_volume(
        #     mesh2,
        #     # generate_triangles=False,
        #     # widget_color=None,
        #     # tubing=False,
        # )

        # ## Some final touches
        # plotter2.background_color = "white"
        # plotter2.view_isometric()
        # plotter2.add_volume_clip_plane(
        #     vol,
        #     normal='-x',
        #     interaction_event='always',
        #     normal_rotation=False,
        # )
        # ## Pass a plotter to stpyvista
        # stpyvista(plotter2)
        
        
        # scout = get_scout(image_np, 'scout')
        # st.image(scout)
    
        # options = ["rect", "transform"]
        # selection = st.pills("Mode", options, selection_mode="single", default = 'rect')
        # # selection = st.selectbox("Mode", options)
        # st.write(selection)
        # canvas_result = st_canvas(
        #     fill_color="#eee",  # Fixed fill color with some opacity
        #     stroke_width=1,
        #     # stroke_color=stroke_color,
        #     # background_color="#eee",
        #     background_image=scout,
        #     update_streamlit=True,
        #     height=200,
        #     drawing_mode=selection,
        #     point_display_radius=0,
        #     key="canvas",
        # )
        
        
    
    # st.write("---")
    

with st.container():

    r2col1, r2col2 = st.columns([3,1], gap="small")
    
    with r2col1:
        
        with stylable_container(
                key="scanner_parameters",
                css_styles=border,
            ):
            st.subheader('Scanner Parameters')
            c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns([1.5,1,1,1.5,1.5,1,1,1,1.5], gap="small")
            
            with c1:
                with stylable_container(
                    key="scan_type_label",
                    css_styles=parameter_label,
                ):
                    st.markdown("**Scan Type**")
                    
                with stylable_container(
                    key="scan_type_key",
                    css_styles=parameter_selectbox,
                ):
                    scan_type = st.selectbox("Scan Type", ['Helical','asdfsf'], label_visibility = "collapsed")

            with c2:
                with stylable_container(
                    key="start_loc_label",
                    css_styles=parameter_label,
                ):
                    st.markdown("**Start**")
                start_loc = st.number_input("Start Location", label_visibility = "collapsed", format="%0.1f")

            with c3:
                with stylable_container(
                    key="end_loc_label",
                    css_styles=parameter_label,
                ):
                    st.markdown("**End**")
                end_loc = st.number_input("End Location", label_visibility = "collapsed", format="%0.1f")

            with c4:
                with stylable_container(
                    key="end_loc_label",
                    css_styles=parameter_label,
                ):
                    st.markdown("**No. Images**")
                num_image = st.number_input("num_of_image",value=0, disabled = True,  label_visibility = "collapsed", format="%d")
                
            with c5:
                with stylable_container(
                    key="gantry_tilt_label",
                    css_styles=parameter_label,
                ):
                    st.markdown("**Gantry Tilt**")
                gantry_tilt = st.number_input("gantry",value=0,  label_visibility = "collapsed", format="%d")
                
            with c6:
                with stylable_container(
                    key="fov_label",
                    css_styles=parameter_label,
                ):
                    st.markdown("**FOV**")
                fov = st.number_input("field of view ",value=0,  label_visibility = "collapsed", format="%d")
            
            with c7:
                with stylable_container(
                    key="kv_label",
                    css_styles=parameter_label,
                ):
                    st.markdown("**kV**")
                kv = st.number_input("kV",value=0,  label_visibility = "collapsed", format="%d")
            
            with c8:
                with stylable_container(
                    key="ma_label",
                    css_styles=parameter_label,
                ):
                    st.markdown("**mA**")
                ma = st.number_input("mA",value=0,  label_visibility = "collapsed", format="%d")
            
            with c9:
                with stylable_container(
                    key="exposure_time_label",
                    css_styles=parameter_label,
                ):
                    st.markdown("**Exp Time**")
                s = st.number_input("exp_time",value=0,  label_visibility = "collapsed", format="%d")
            
            add_vertical_space(5)
            c1, c2, c3, c4, c5 = st.columns([1.5,2,1.5,4.5,1.5], gap="small")
            with c1:
                with stylable_container(
                    key="end_exam",
                    css_styles="""
                        button {
                            background-color: red;
                            color: white;
                            border-radius: 20px;
                        }
                        """,
                ):
                    st.button("**End Exam**")
                    
            with c2:
                with stylable_container(
                        key="new_protocol",
                        css_styles=parameter_button,
                ):
                    st.button("**Select New Protocol**")
            with c3:
                with stylable_container(
                        key="next_series",
                        css_styles=parameter_button,
                ):
                    st.button("**Next Series**")
            with c5:
                with stylable_container(
                        key="start_scan",
                        css_styles="""
                        button {
                            background-color: green;
                            color: white;
                            border-radius: 20px;
                        }
                        """,
                ):
                    st.button("**Start Scan**")
                    
    with r2col2:
        
        with stylable_container(
                key="scanner_parameters",
                css_styles=border,
            ):
            tab1, tab2, tab3 = st.tabs(["**Imaging**", "**Thickness**", "**Pr. Injector**"])
            with tab1:
                img_grid = grid(2, 2, 2, 2, vertical_align="bottom")
                
                img_grid.selectbox('**Image Matrix**', options=['512x512', '256x256', '128x128'], key='matrix_size')
                img_grid.selectbox('**Plane**', options=['Axial', 'Coronal', 'Sagital'], key='plane')
                img_grid.selectbox('**Reconstruction Algorithm**', options=['Sharp', 'Smooth'], key='recon_alg')
                img_grid.selectbox('**Filter**', options=['Gaussian', 'Butterwroth'], key='filter')
                img_grid.number_input('**Contrast Amount**', key='contrast_amount')
                img_grid.text_input('**Contrast Agent**', key='contrast_agent')
                img_grid.number_input('**Window width**', key='ww', min_value=0, value=80)
                img_grid.number_input('**Window level**', key='wl', min_value=-1000, value=0)