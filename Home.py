import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.sidebar.success("Select a demo above.")

st.write("# Welcome to NMLAB Applications! 👋")

st.markdown(
    """
    **👈 Select an application from the sidebar** 
    - Image filtering
    - Image reconstruction in Nuclear Medicine
    - Profile, Projection and Sinogram
"""
)

st.divider()
st.caption("Anucha Chaichana") 
st.caption("anucha.cha@mahidol.ac.th")