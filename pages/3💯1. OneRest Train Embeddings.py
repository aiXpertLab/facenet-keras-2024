import streamlit as st, os
from streamlit_extras.stateful_button import button

from utils import streamlit_components
from apps import one_dataset, one_embedding

streamlit_components.streamlit_ui('🦣 Face Classification - Embeddings and Dataset')

model      = os.getenv('FACENET_MODEL')
dataset    = os.getenv('ONE_DATASET_TRAINING')
embeddings = os.getenv('ONE_EMBEDDINGS_TRAINING')

face_folder= os.getenv('ONE_TRAIN_FOLDER')

st.text(dataset)
st.text(face_folder)
st.text(embeddings)

if button("Save One Dataset and Embeddings?", key="but31"):
    one_dataset.save_dataset(dataset=dataset, face_folder = face_folder)
    one_embedding.save_embeddings(model_name=model, embedding_name=embeddings, dataset_name=dataset)
    