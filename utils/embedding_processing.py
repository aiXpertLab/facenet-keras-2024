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


def save_embeddings(model, embedding_name, datasize_name):
    with st.spinner('embedding...'):
        data = numpy.load(datasize_name)
        trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        st.write('Loaded: ', trainX.shape,
                 trainy.shape, testX.shape, testy.shape)
        model = tf.keras.models.load_model(model)
        # convert each face in the train set to an embedding
        newTrainX = list()
        for face_pixels in trainX:
            embedding = get_embedding(model, face_pixels)
            newTrainX.append(embedding)
        newTrainX = numpy.asarray(newTrainX)
        print(newTrainX.shape)
        # convert each face in the test set to an embedding
        newTestX = list()
        for face_pixels in testX:
            embedding = get_embedding(model, face_pixels)
            newTestX.append(embedding)
        newTestX = numpy.asarray(newTestX)
        print(newTestX.shape)
        # save arrays to one file in compressed format
        numpy.savez_compressed(embedding_name, newTrainX,
                               trainy, newTestX, testy)
    st.success(f"saved to:  {embedding_name}")
