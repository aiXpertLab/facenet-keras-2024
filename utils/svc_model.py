import numpy
import random
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_extras.stateful_button import button

from sklearn import preprocessing
from sklearn.svm import SVC

from utils.dataset_processing import load_dataset_with_metadata


def show_svc(model, testX, testX_faces, out_encoder):
    # Pick another random image
    selection = random.choice([i for i in range(testX.shape[0])])
    random_face_pixels = testX_faces[selection]
    random_face_emb = testX[selection]
    # random_face_class = testy[selection]
    # random_face_name = out_encoder.inverse_transform([random_face_class])

    # Prediction for the face
    samples = numpy.expand_dims(random_face_emb, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)

    # Get prediction details
    class_index = yhat_class[0]
    class_probability = yhat_prob[0, class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    
    # Display prediction
    st.write(f'Predicted based on model: {predict_names[0]} ({class_probability:.3f})')
    # st.write(f'Expected based on label: {random_face_name[0]}')

    # Update displayed image
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(random_face_pixels)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    # ax.set_title(title, fontsize=6) 
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    st.pyplot(fig)


def svc(DATASIZE_NAME, EMBEDDINGS_NAME):
    # with st.spinner('loading ...'):    
        # Load faces
        
        
        data = numpy.load(DATASIZE_NAME)
        testX_faces = data['arr_2']
        
        # Load face embeddings
        data = numpy.load(EMBEDDINGS_NAME)
        trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

        # Normalize input vectors
        in_encoder = preprocessing.Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)

        # Label encode targets
        out_encoder = preprocessing.LabelEncoder()
        out_encoder.fit(trainy)
        trainy = out_encoder.transform(trainy)
        # testy = out_encoder.transform(testy)

        # Fit model
        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainy)

        show_svc(model, testX, testX_faces, out_encoder)
