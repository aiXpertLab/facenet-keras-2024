import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import PIL
import numpy
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mtcnn.mtcnn import MTCNN

detector = MTCNN()


def load_image(filename):
    image = PIL.Image.open(filename)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pixels = numpy.asarray(image)
    return pixels


def detect_image(pixels, required_size=(160, 160)):
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


def load_faces_from_train_val_prod(directory):
    # Train, test, prod
    X, y = list(), list()
    for subdir in os.listdir(directory):
        # path = directory + subdir + '/'
        path = os.path.join(directory, subdir)
        if not os.path.isdir(path):
            continue

        faces = load_faces_from_one_directory(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        st.write(labels)
        st.write(f"> Loaded {len(faces)} examples for class: {subdir}")
        # store
        X.extend(faces)
        y.extend(labels)
    return numpy.asarray(X), numpy.asarray(y)


def extract_face(filename, required_size=(160, 160)):
    pixels = load_image(filename)
    face_array = detect_image(pixels)
    return face_array

def is_image_file(filename):
    return filename.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))

def show_extracted_faces(folder: str) -> None:
    with st.spinner('Extracting...'):
        i = 1
        cols = st.columns(7)
        
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            
            if os.path.isfile(path) and is_image_file(filename):
            
                try:
                    face = extract_face(path)
                    with cols[(i - 1) % 7]:
                        fig, ax = plt.subplots()
                        ax.axis('off')
                        ax.imshow(face)
                        st.pyplot(fig)
                    i += 1
                    if i % 7 == 1:
                        cols = st.columns(7)
                except Exception as e:
                    st.error(f"Error processing {filename}: {e}")
                
    st.success('Done')

# Example usage:
# face_processor = FaceProcessor()
# face_processor.draw_image_with_boxes('path/to/image.jpg', results)
# face_processor.show_img(images, image_names)
# face_processor.show_extracted_faces('path/to/folder/')

def load_faces_prod(directory):
    # Train, test, prod
    X, y = list(), list()
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if not os.path.isdir(path):
            continue

        faces = load_faces_from_one_directory(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        st.write(labels)
        st.write(f"> Loaded {len(faces)} examples for class: {subdir}")
        # store
        X.extend(faces)
        y.extend(labels)
    return numpy.asarray(X), numpy.asarray(y)


def load_faces_with_path(directory):
    # Train, test, prod
    X, y, file_names = list(), list(), list()
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if not os.path.isdir(path):
            continue

        faces, file_names = load_faces_from_one_directory(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        st.write(labels)
        st.write(f"> Loaded {len(faces)} examples for class: {subdir}")
        # store
        X.extend(faces)
        y.extend(labels)
        # file_names.extend(file_names)
    return numpy.asarray(X), numpy.asarray(y), file_names


def load_faces_from_one_directory(directory):
    faces = list()
    file_names = list()
    # enumerate files
    for filename in os.listdir(directory):
        # path = directory + filename
        if is_image_file(filename):
            path = os.path.join(directory, filename)
            face = extract_face(path)
            if face is not None:
                faces.append(face)      # detected faces
                file_names.append(filename)
            st.info(f'faces: {path} {len(file_names)}')
    return faces, file_names

