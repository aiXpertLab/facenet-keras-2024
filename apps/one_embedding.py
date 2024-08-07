import numpy
import tensorflow as tf

import streamlit as st
from streamlit_extras.stateful_button import button

from utils.face_processing import load_faces_from_one_directory


# get the face embedding for one face
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')    # scale pixel values
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = numpy.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)    # make prediction to get embedding
    return yhat[0]


def save_embeddings(model_name, embedding_name, dataset_name):
    # one set dataset
    with st.spinner('embedding...'):
        data = numpy.load(dataset_name)
        trainX, trainy = data['arr_0'], data['arr_1']
        st.write('Loaded: ', trainX.shape, trainy.shape)
        model = tf.keras.models.load_model(model_name)
        # convert each face in the train set to an embedding
        newTrainX = list()
        for face_pixels in trainX:
            embedding = get_embedding(model, face_pixels)
            newTrainX.append(embedding)
        newTrainX = numpy.asarray(newTrainX)
        print(newTrainX.shape)
        # convert each face in the test set to an embedding
        # save arrays to one file in compressed format
        numpy.savez_compressed(embedding_name, newTrainX, trainy)
    st.success(f"saved to:  {embedding_name}")
