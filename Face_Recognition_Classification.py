import streamlit as st
from utils import streamlit_components

from dotenv import load_dotenv
load_dotenv()


streamlit_components.streamlit_ui('🐬🦣 Face Recognition 🍃🦭')


tab1, tab2 = st.tabs(["General","MTCNN",])

with tab1: streamlit_components.general()
with tab2: streamlit_components.MTCNN()   
