import streamlit as st, os

from streamlit_extras.stateful_button import button
from utils import delivering_classification, streamlit_components, image_processing, face_processing 

streamlit_components.streamlit_ui('🦣 Face Classification')

dataset_prod        = os.getenv('PROD_DATASET_ywsd')
dataset_training    = os.getenv('TRAINING_DATASET_ywsd')

embeddings_prod     = os.getenv('PROD_EMBEDDINGS_ywsd')
embeddings_training = os.getenv('TRAINING_EMBEDDINGS_ywsd')

model      = os.getenv('FACENET_MODEL')


if button("Continue ?", key="button4"): 
    delivering_classification.svc(dataset_training, embeddings_training)
        
