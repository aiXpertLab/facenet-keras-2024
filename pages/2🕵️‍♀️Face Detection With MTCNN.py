import streamlit as st
from streamlit_extras.stateful_button import button
from utils import streamlit_components, image_processing as ip
streamlit_components.streamlit_ui('🦣 Face Detection With MTCNN')
# -------------------------------------------------------------------------------------
import os
from matplotlib import pyplot
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

directory = os.getenv('DATA_IMAGES')

if button("Continue?", key="butt2"):
    for filename in os.listdir(directory):
        
        path = directory + filename
        st.text(path)

        pixels = pyplot.imread(path)# load the photograph
        detector = MTCNN()

        faces = detector.detect_faces(pixels)

        ip.draw_image_with_boxes(filename=path, result_list=faces)