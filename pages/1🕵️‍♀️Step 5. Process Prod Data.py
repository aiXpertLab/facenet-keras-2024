import streamlit as st, os

from streamlit_extras.stateful_button import button
from utils import streamlit_components, face_pipline, image_processing, face_processing 
from utils import dataset_processing, streamlit_components, image_processing, face_processing 
from utils import streamlit_components, face_pipline, image_processing, embedding_processing 


streamlit_components.streamlit_ui('🦣 Face Classification - Embeddings and Dataset')

dataset    = os.getenv('PROD_DATASET_ywsd')
embeddings = os.getenv('PROD_EMBEDDINGS_ywsd')
model      = os.getenv('FACENET_MODEL')

st.text(dataset)

if button("Save Production Dataset and Embeddings?", key="but5"):
    dataset_processing.save_dataset(dataset, os.getenv('TRAIN'), os.getenv('VAL'))
    embedding_processing.save_embeddings(model, embeddings, dataset)
    