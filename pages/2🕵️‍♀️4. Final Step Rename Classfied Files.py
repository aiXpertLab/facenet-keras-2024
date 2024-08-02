import streamlit as st, os
from streamlit_extras.stateful_button import button

import matplotlib.pyplot as plt

from utils import delivering_classification, streamlit_components, image_processing

streamlit_components.streamlit_ui('🦣 Face Classification')

dataset_prod        = os.getenv('PROD_DATASET_ywsd')

embeddings_prod     = os.getenv('PROD_EMBEDDINGS_ywsd')
embeddings_training = os.getenv('TRAINING_EMBEDDINGS_ywsd')

model = os.getenv('FACENET_MODEL')
PROD  = os.getenv('PENDING')

if button("Rename?", key="button24"): 
    predictions = delivering_classification.classify_all_images(embeddings_training, dataset_prod, embeddings_prod)

    with st.spinner('renaming ...'):    
    
        for i in range(len(predictions)):
            name, prob, face_pixels, file_name = predictions[i]
            
            image_file = PROD + file_name[i]
            new_file = PROD + name + '_' + file_name[i]
            st.write(f'Predicted: {name} ({prob:.3f}%)')

            if prob > 90:
                os.rename(image_file, new_file)
            
            # image_processing.draw_image(image_file)
            # fig, ax = plt.subplots(figsize=(2, 2))
            # ax.imshow(face_pixels)
            # ax.set_xticks([])  # Remove x-axis ticks
            # ax.set_yticks([])  # Remove y-axis ticks
            # st.pyplot(fig)
            
    st.success('done!')