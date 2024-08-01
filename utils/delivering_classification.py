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


def svc_matching(embeddings_training, dataset_prod, embeddings_prod):   #fitted_model, testX, testX_faces, out_encoder)
    # Pick another random image
    fitted_model, out_encoder = svc_fitting(embeddings_training)
    
    selection = random.choice([i for i in range(testX.shape[0])])   # 100th picture
    random_face_pixels = testX_faces[selection]
    random_face_emb = testX[selection]

    # Prediction for the face
    samples = numpy.expand_dims(random_face_emb, axis=0)
    yhat_class = fitted_model.predict(samples)
    yhat_prob  = fitted_model.predict_proba(samples)

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


# def classify_all_images(model, encoder, testX, testX_faces):
def classify_all_images(embeddings_training, dataset_prod, embeddings_prod):

    fitted_model, out_encoder = svc_fitting(embeddings_training)

    dataset_prod = numpy.load(dataset_prod)
    testX_faces = dataset_prod['arr_0']

    # Load face embeddings
    data_train = numpy.load(embeddings_training)
    data_prod  = numpy.load(embeddings_prod)
    trainX,trainy = data_train['arr_0'], data_train['arr_1']
    testX,  testy = data_prod['arr_0'], data_prod['arr_1']

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

