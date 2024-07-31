import streamlit as st
from streamlit_extras.stateful_button import button
from utils import streamlit_components, image_processing as ip
streamlit_components.streamlit_ui('🦣 Show Training Pictures')
# -------------------------------------------------------------------------------------
import os
from matplotlib import pyplot

# directory = os.getenv('DATA_IMAGES')

if button("Show Training Data?", key="butt3"):
        
    directory = os.getenv('YCC')
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg', 'webp'))]

    # Initialize session state for pagination
    if 'page' not in st.session_state:
        st.session_state.page = 0

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
        ip.draw_image(filename=path)
                

    # Pagination buttons
    if st.session_state.page > 0:
        if st.button('Previous'):
            st.session_state.page -= 1
            st.experimental_rerun()

    if st.session_state.page < total_pages - 1:
        if st.button('Next'):
            st.session_state.page += 1
            st.experimental_rerun()