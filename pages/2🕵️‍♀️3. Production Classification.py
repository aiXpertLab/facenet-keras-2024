import streamlit as st, os
from streamlit_extras.stateful_button import button

import matplotlib.pyplot as plt

from utils import delivering_classification, streamlit_components

streamlit_components.streamlit_ui('🦣 Face Classification')

dataset_prod        = os.getenv('PROD_DATASET_ywsd')

embeddings_prod     = os.getenv('PROD_EMBEDDINGS_ywsd')
embeddings_training = os.getenv('TRAINING_EMBEDDINGS_ywsd')

model      = os.getenv('FACENET_MODEL')

if button("Production Classification", key="button23"): 
    predictions = delivering_classification.classify_all_images(embeddings_training, dataset_prod, embeddings_prod)
    # Display results
    # for predict_name, class_probability, face_pixels in predictions:
    #     st.write(f'Predicted: {predict_name} ({class_probability:.3f}%)')
        
    #     fig, ax = plt.subplots(figsize=(1, 1))
    #     ax.imshow(face_pixels)
    #     ax.set_xticks([])  # Remove x-axis ticks
    #     ax.set_yticks([])  # Remove y-axis ticks
    #     st.pyplot(fig)

    # Pagination
    items_per_page = 5
    total_items = len(predictions)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    page = st.sidebar.slider('Select Page', 1, total_pages, 1)
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    st.write(f'Page {page} of {total_pages}')
    
    for i in range(start_idx, end_idx):
        name, prob, face_pixels = predictions[i]
        st.write(f'Predicted: {name} ({prob:.3f}%)')
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(face_pixels)
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        st.pyplot(fig)
        
