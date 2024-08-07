import streamlit as st, os
from streamlit_extras.stateful_button import button
from utils import streamlit_components, face_pipline, image_processing, face_processing 

FACENET_MODEL   = os.getenv('FACENET_MODEL')
DATASIZE_NAME   = os.getenv('DATASET_NAME')
EMBEDDINGS_NAME = os.getenv('EMBEDDINGS_NAME')

streamlit_components.streamlit_ui('🦣 Classification with FaceNet')
t1,t2,t3,t4 = st.tabs(["Face Classification", "Extract Faces","Save Dataset","Embeddings",])

with t2: 
    if button("Extract Face", key="button1"):
        face_processing.show_extracted_faces(os.getenv('YCC'))

with t3:
    if button("Save Datasize", key="button2"):
        face_pipline.save_datasize_train_val(DATASIZE_NAME, os.getenv('TRAIN'), os.getenv('VAL'))

with t4:
    if button("Save Embeddings", key="button3"):
        face_pipline.save_embeddings(FACENET_MODEL, EMBEDDINGS_NAME, DATASIZE_NAME)
        
with t1:
    if button("Pick One?", key="button4"): 
        face_pipline.svc(DATASIZE_NAME, EMBEDDINGS_NAME)
        