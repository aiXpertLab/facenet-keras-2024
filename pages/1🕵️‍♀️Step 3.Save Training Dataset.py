import streamlit as st, os
from streamlit_extras.stateful_button import button
from utils import dataset_processing, streamlit_components, image_processing, face_processing 

streamlit_components.streamlit_ui('🦣 Save Dataset')

dataset    = os.getenv('TRAINING_DATASET_ywsd')

st.text(dataset)

if button("Save Training Dataset?", key="button2"):
    dataset = dataset_processing.save_dataset(dataset, os.getenv('TRAIN'), os.getenv('VAL'))
    if dataset:
        st.success(f"saved to:  {dataset}")
    else:
        st.error(f"check existence of {dataset}")
