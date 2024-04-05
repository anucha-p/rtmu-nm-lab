import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.sidebar.success("Select a demo above.")

st.write("# Welcome to NMLAB Demo Application! 👋")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some example!
    ### Want to learn more?
    - Profile, Projection and Sinogram
    - Image filtering
    - Image reconstruction in Nuclear Medicine
"""
)

st.divider()
st.caption("Anucha Chaichana") 
st.caption("anucha.cha@mahidol.edu")

