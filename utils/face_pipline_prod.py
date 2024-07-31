import os
import PIL
import random
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from streamlit_extras.stateful_button import button
import logging

# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file named "app.log"
        logging.StreamHandler()          # Also log to the console
    ]
)

class FaceProcessor:
    def __init__(self, required_size=(160, 160)):
        self.required_size = required_size
        self.detector = MTCNN()
        logging.info("FaceProcessor initialized with required size: %s", required_size)

    def load_image(self, filename):
        logging.info("Loading image from file: %s", filename)
        image = PIL.Image.open(filename)
        image = image.convert('RGB')
        pixels = np.asarray(image)
        return pixels

    def detect_image(self, pixels):
        logging.info("Detecting face in image")
        results = self.detector.detect_faces(pixels)
        if len(results) == 0:
            logging.warning("No faces detected")
            return None
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = PIL.Image.fromarray(face)
        image = image.resize(self.required_size)
        face_array = np.asarray(image)
        return face_array

    def extract_face(self, filename):
        logging.info("Extracting face from file: %s", filename)
        pixels = self.load_image(filename)
        face_array = self.detect_image(pixels)
        return face_array

class DatasetLoader:
    def __init__(self, face_processor):
        self.face_processor = face_processor

    def load_faces(self, directory):
        logging.info("Loading faces from directory: %s", directory)
        faces = []
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            face = self.face_processor.extract_face(path)
            if face is not None:
                faces.append(face)
        return faces

    def load_dataset(self, directory):
        logging.info("Loading dataset from directory: %s", directory)
        X, y = [], []
        for subdir in os.listdir(directory):
            path = os.path.join(directory, subdir)
            if not os.path.isdir(path):
                continue
            faces = self.load_faces(path)
            labels = [subdir] * len(faces)
            logging.info("Loaded %d examples for class: %s", len(faces), subdir)
            X.extend(faces)
            y.extend(labels)
        return np.asarray(X), np.asarray(y)

class ModelTrainer:
    def __init__(self, model_path):
        self.model_path = model_path
        logging.info("ModelTrainer initialized with model path: %s", model_path)

    def get_embedding(self, model, face_pixels):
        logging.info("Getting embedding for face")
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = np.expand_dims(face_pixels, axis=0)
        yhat = model.predict(samples)
        return yhat[0]

    def save_datasize(self, datasize_name, train, val):
        if os.path.exists(datasize_name):
            if button("Existed. Override?", key="fp1"):
                with st.spinner('saving ...'):
                    trainX, trainy = DatasetLoader(FaceProcessor()).load_dataset(train)
                    testX, testy = DatasetLoader(FaceProcessor()).load_dataset(val)
                    np.savez_compressed(datasize_name, trainX, trainy, testX, testy)
                logging.info("Saved dataset to: %s", datasize_name)
                st.success(f"Saved to: {datasize_name}")
        else:
            with st.spinner('saving ...'):
                trainX, trainy = DatasetLoader(FaceProcessor()).load_dataset(train)
                testX, testy = DatasetLoader(FaceProcessor()).load_dataset(val)
                np.savez_compressed(datasize_name, trainX, trainy, testX, testy)
            logging.info("Saved dataset to: %s", datasize_name)
            st.success(f"Saved to: {datasize_name}")

    def save_embeddings(self, embedding_name, datasize_name):
        with st.spinner('embedding...'):
            data = np.load(datasize_name)
            trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
            logging.info('Loaded dataset shapes: trainX: %s, trainy: %s, testX: %s, testy: %s',
                         trainX.shape, trainy.shape, testX.shape, testy.shape)
            model = tf.keras.models.load_model(self.model_path)
            newTrainX = np.asarray([self.get_embedding(model, face_pixels) for face_pixels in trainX])
            newTestX = np.asarray([self.get_embedding(model, face_pixels) for face_pixels in testX])
            np.savez_compressed(embedding_name, newTrainX, trainy, newTestX, testy)
            logging.info("Saved embeddings to: %s", embedding_name)
        st.success(f"Saved to: {embedding_name}")

class SVCModel:
    def __init__(self):
        self.model = SVC(kernel='linear', probability=True)
        logging.info("SVCModel initialized with linear kernel")

    def show_svc(self, model, testX, testX_faces, testy, out_encoder):
        selection = random.choice([i for i in range(testX.shape[0])])
        random_face_pixels = testX_faces[selection]
        random_face_emb = testX[selection]
        random_face_class = testy[selection]
        random_face_name = out_encoder.inverse_transform([random_face_class])
        samples = np.expand_dims(random_face_emb, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        logging.info('Prediction: %s (%.3f%%), Expected: %s',
                     predict_names[0], class_probability, random_face_name[0])
        st.write(f'Predicted: {predict_names[0]} ({class_probability:.3f}%)')
        st.write(f'Expected: {random_face_name[0]}')
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.imshow(random_face_pixels)
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)

    def train_and_show(self, datasize_name, embeddings_name):
        with st.spinner('loading ...'):
            data_faces = np.load(datasize_name)
            testX_faces = data_faces['arr_2']
            data_emb = np.load(embeddings_name)
            trainX, trainy, testX, testy = data_emb['arr_0'], data_emb['arr_1'], data_emb['arr_2'], data_emb['arr_3']
            in_encoder = preprocessing.Normalizer(norm='l2')
            trainX = in_encoder.transform(trainX)
            testX = in_encoder.transform(testX)
            out_encoder = preprocessing.LabelEncoder()
            out_encoder.fit(trainy)
            trainy = out_encoder.transform(trainy)
            testy = out_encoder.transform(testy)
            self.model.fit(trainX, trainy)
            logging.info("SVC model trained")
        self.show_svc(self.model, testX, testX_faces, testy, out_encoder)

# Example usage
if __name__ == "__main__":
    face_processor = FaceProcessor()
    dataset_loader = DatasetLoader(face_processor)
    model_trainer = ModelTrainer('model_path.h5')
    svc_model = SVCModel()

    # Replace 'train', 'val', 'datasize_name', 'embedding_name' with actual paths
    model_trainer.save_datasize('datasize_name.npz', 'train/', 'val/')
    model_trainer.save_embeddings('embedding_name.npz', 'datasize_name.npz')
    svc_model.train_and_show('datasize_name.npz', 'embedding_name.npz')
