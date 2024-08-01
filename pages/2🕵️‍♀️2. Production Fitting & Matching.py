import streamlit as st, os

from streamlit_extras.stateful_button import button
from utils import streamlit_components, svc_model, image_processing, face_processing 

streamlit_components.streamlit_ui('🦣 Face Classification')

dataset_prod        = os.getenv('PROD_DATASET_ywsd')
dataset_training    = os.getenv('TRAINING_DATASET_ywsd')

embeddings_prod     = os.getenv('PROD_EMBEDDINGS_ywsd')
embeddings_training = os.getenv('TRAINING_EMBEDDINGS_ywsd')

model      = os.getenv('FACENET_MODEL')


if button("Production", key="button22"): 
    svc_model.svc_matching(dataset_training, embeddings_training, dataset_prod, embeddings_prod)
        
