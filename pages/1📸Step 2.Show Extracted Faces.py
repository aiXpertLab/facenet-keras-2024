import streamlit as st, os
from streamlit_extras.stateful_button import button
from utils import streamlit_components, face_pipline, image_processing, face_processing 

FACENET_MODEL   = os.getenv('FACENET_MODEL')
DATASIZE_NAME   = os.getenv('DATASIZE_NAME')
EMBEDDINGS_NAME = os.getenv('EMBEDDINGS_NAME')

streamlit_components.streamlit_ui('🦣 Classification with FaceNet')

if button("Extract Face", key="button1"):
    face_processing.show_extracted_faces(os.getenv('YCC'))
