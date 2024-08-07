import streamlit as st, os, shutil
from streamlit_extras.stateful_button import button

import matplotlib.pyplot as plt

from utils import streamlit_components, image_processing
from apps import one_classify

streamlit_components.streamlit_ui('🦣 Face Classification')

dataset_prod        = os.getenv('ONE_DATASET_TESTING')

embeddings_prod     = os.getenv('ONE_EMBEDDINGS_TESTING')
embeddings_training = os.getenv('ONE_EMBEDDINGS_TRAINING')

model = os.getenv('FACENET_MODEL')
ONE_TESTING_FOLDER  = os.getenv('ONE_TESTING_FOLDER')
destination_dir  = os.getenv('ONE_TESTING_SUBFOLDER')

if button("Move false to sub?", key="but331"): 
    predictions = one_classify.classify_all_images(embeddings_training, dataset_prod, embeddings_prod)
    os.makedirs(destination_dir, exist_ok=True)

    for i in range(len(predictions)):

        name, prob, face_pixels, file_name = predictions[i]
        st.write(file_name[i])
        # st.write(PROD)
        
        image_file = ONE_TESTING_FOLDER + file_name[i]
        
        # image_files = os.path.join(PROD, file_name)
        
        if prob > 99.9999999:
            destination_file = os.path.join(destination_dir, file_name[i])
            shutil.move(image_file, destination_file)
            st.text(f'Predicted: {name} ({prob:.6f}%)')
        # image_processing.draw_image(image_file)


    if button("Pagenation?", key="but332"): 
        # Pagination
        items_per_page = 5
        total_items = len(predictions)
        total_pages = (total_items + items_per_page - 1) // items_per_page
        
        page = st.sidebar.slider('Select Page', 1, total_pages, 1)
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        st.write(f'Page {page} of {total_pages}')
        
        for i in range(start_idx, end_idx):

            name, prob, face_pixels, file_name = predictions[i]
            st.write(file_name[i])
            # st.write(PROD)
            
            image_file = ONE_TESTING_FOLDER + file_name[i]
            
            # image_files = os.path.join(PROD, file_name)
            
            if prob < 99:
                st.error(f'Predicted: {name} ({prob:.3f}%)')                    
            else:
                st.text(f'Predicted: {name} ({prob:.3f}%)')
            image_processing.draw_image(image_file)
            
            # fig, ax = plt.subplots(figsize=(2, 2))
            # ax.imshow(face_pixels)
            # ax.set_xticks([])  # Remove x-axis ticks
            # ax.set_yticks([])  # Remove y-axis ticks
            # st.pyplot(fig)
            
