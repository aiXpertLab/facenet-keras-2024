import os
import PIL
import random
import streamlit as st
import numpy
from sklearn import preprocessing
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from streamlit_extras.stateful_button import button

import tensorflow as tf
from mtcnn.mtcnn import MTCNN


def load_image(filename):
    image = PIL.Image.open(filename)
    image = image.convert('RGB')
    pixels = numpy.asarray(image)
    return pixels


def detect_image(pixels, required_size=(160, 160)):
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    if len(results) == 0:
        # No faces detected
        return None
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = PIL.Image.fromarray(face)
    image = image.resize(required_size)
    face_array = numpy.asarray(image)
    return face_array


def extract_face(filename, required_size=(160, 160)):
    pixels = load_image(filename)
    face_array = detect_image(pixels)
    return face_array

# load images and extract faces for all images in a directory


def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in os.listdir(directory):
        path = directory + filename
        face = extract_face(path)
        if face is not None:
            faces.append(face)      # detected faces
    return faces

# load a dataset that contains one subdir for each class that in turn contains images


def load_dataset(directory):
    X, y = list(), list()
    for subdir in os.listdir(directory):
        path = directory + subdir + '/'
        if not os.path.isdir(path):
            continue

        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        st.write(labels)
        st.write(f"> Loaded {len(faces)} examples for class: {subdir}")
        # store
        X.extend(faces)
        y.extend(labels)
    return numpy.asarray(X), numpy.asarray(y)

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


def save_datasize(datasize_name, train, val):
    if os.path.exists(datasize_name):
        if button("Existed. Overide?", key="fp1"):
            with st.spinner('saving ...'):
                trainX, trainy = load_dataset(train)
                testX, testy = load_dataset(val)
                # save arrays to one file in compressed format
                numpy.savez_compressed(
                    datasize_name, trainX, trainy, testX, testy)
            st.success(f"saved to:  {datasize_name}")
    else:
        with st.spinner('saving ...'):
            trainX, trainy = load_dataset(train)
            testX, testy = load_dataset(val)
            # save arrays to one file in compressed format
            numpy.savez_compressed(datasize_name, trainX, trainy, testX, testy)
        st.success(f"saved to:  {datasize_name}")


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

def show_svc(model, testX, testX_faces, testy, out_encoder):
    # Pick another random image
    selection = random.choice([i for i in range(testX.shape[0])])
    random_face_pixels = testX_faces[selection]
    random_face_emb = testX[selection]
    random_face_class = testy[selection]
    random_face_name = out_encoder.inverse_transform([random_face_class])

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
    st.write(f'Expected based on label: {random_face_name[0]}')

    # Update displayed image
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(random_face_pixels)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    # ax.set_title(title, fontsize=6) 
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    st.pyplot(fig)



def svc(DATASIZE_NAME, EMBEDDINGS_NAME):
    with st.spinner('loading ...'):    
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
        testy = out_encoder.transform(testy)

        # Fit model
        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainy)

    show_svc(model, testX, testX_faces, testy, out_encoder)

