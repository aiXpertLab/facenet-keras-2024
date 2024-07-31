import streamlit as st, os
from streamlit_extras.stateful_button import button
from utils import streamlit_components, face_pipline_prod, image_processing, face_processing

FACENET_MODEL   = os.getenv('FACENET_MODEL')
DATASIZE_ywsd   = os.getenv('DATASIZE_NAME')
EMBEDDINGS_ywsd = os.getenv('EMBEDDINGS_NAME')

streamlit_components.streamlit_ui('🦣 Classification with FaceNet')
t1,t2,t3,t4,t5 = st.tabs(["Prepare 1: Show Original Pictures", "Prepare 2. Show Extract Faces", "Save Training Dataset","Embeddings","Face Classification", ])

with t1: 
    
    if 'page' not in st.session_state: st.session_state.page = 0
    total_pages = 0

    if button("Show Original Pictures?", key="button1"):
        directory = os.getenv('YCC')
        image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg', 'webp'))]

        # Initialize session state for pagination

        # Number of images per page
        images_per_page = 3

        # Calculate the total number of pages
        total_pages = (len(image_files) + images_per_page - 1) // images_per_page

        # Get the images for the current page
        start_index = st.session_state.page * images_per_page
        end_index = start_index + images_per_page
        current_images = image_files[start_index:end_index]

        # Display the images for the current page
        for path in current_images:
            st.text(path)
            image_processing.draw_image(filename=path)
                    
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.page > 0:
            if st.button('Previous'):
                st.session_state.page -= 1
                st.experimental_rerun()

    with col2:
        if st.session_state.page < total_pages - 1:
            if st.button('Next'):
                st.session_state.page += 1
                st.experimental_rerun()


with t2: 
    if button("Extract Faces", key="button2"):
        face_processing.show_extracted_faces(os.getenv('TRAIN'))
        
# with t2:
#     if button("Save Training Datasize", key="button2"):
#         face_pipline_prod.save_datasize(DATASIZE_NAME, os.getenv('TRAIN'), os.getenv('VAL'))

# with t4:
#     if button("Save Embeddings", key="button3"):
#         face_pipline_prod.save_embeddings(FACENET_MODEL, EMBEDDINGS_NAME, DATASIZE_NAME)
        
# with t3:
#     if button("Pick One?", key="button4"): 
#         face_pipline_prod.svc(DATASIZE_NAME, EMBEDDINGS_NAME)
        
