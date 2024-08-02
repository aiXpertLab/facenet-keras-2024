import numpy 
import matplotlib.pyplot as plt

import streamlit as st, os
from streamlit_extras.stateful_button import button

from sklearn import preprocessing
from sklearn.svm import SVC

from utils.dataset_processing import load_dataset_with_metadata

def svc_fitting(embeddings_training):

        # Load face embeddings
        data_train = numpy.load(embeddings_training)
        trainX,trainy = data_train['arr_0'], data_train['arr_1']
        
        # Normalize input vectors
        in_encoder = preprocessing.Normalizer(norm='l2')
        trainX= in_encoder.transform(trainX)

        # Label encode targets
        out_encoder = preprocessing.LabelEncoder()
        out_encoder.fit(trainy)
        trainy= out_encoder.transform(trainy)
        # st.write(data_train['arr_0'].shape)  #  (164, 128)
        # st.write(data_train['arr_1'].shape)  # This should be (164,)

        # Fit model
        SVC_model = SVC(kernel='linear', probability=True)
        SVC_model.fit(trainX, trainy)

        st.write(f"Number of Support Vectors: {SVC_model.n_support_.sum()}")
        st.write(f"Support Vectors per Class: {SVC_model.n_support_}")
        # st.write(f"Model Coefficients: {SVC_model.coef_}")
        st.write(f"Model Intercept: {SVC_model.intercept_}")
    
        return SVC_model, out_encoder


# def classify_all_images(model, encoder, testX, testX_faces):
def classify_all_images(embeddings_training, dataset_prod, embeddings_prod):

    fitted_model, out_encoder = svc_fitting(embeddings_training)

    dataset_prod = numpy.load(dataset_prod)
    testX_faces = dataset_prod['arr_0']

    # Load face embeddings
    data_prod  = numpy.load(embeddings_prod)
    testX = data_prod['arr_0']

    predictions = []
    
    for i in range(testX.shape[0]):
        face_emb = testX[i]
        face_pixels = testX_faces[i]
        
        # Make a prediction
        samples = numpy.expand_dims(face_emb, axis=0)
        yhat_class = fitted_model.predict(samples)
        yhat_prob = fitted_model.predict_proba(samples)
        
        # Get prediction details
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_name = out_encoder.inverse_transform(yhat_class)[0]
        
        # Store the result
        predictions.append((predict_name, class_probability, face_pixels))
    
    return predictions
