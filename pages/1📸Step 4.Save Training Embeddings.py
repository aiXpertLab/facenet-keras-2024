import streamlit as st, os
from streamlit_extras.stateful_button import button
from utils import streamlit_components, face_pipline, image_processing, embedding_processing 


streamlit_components.streamlit_ui('🦣 Embeddings')

dataset    = os.getenv('TRAINING_DATASET_ywsd')
embeddings = os.getenv('TRAINING_EMBEDDINGS_ywsd')
model      = os.getenv('FACENET_MODEL')

if button("Save Training Embeddings", key="button3"):
    embedding_processing.save_embeddings(model, embeddings, dataset)
    