import streamlit as st
import numpy, os, PIL

from mtcnn.mtcnn import MTCNN
detector = MTCNN()


def save_dataset(dataset, face_folder):
    with st.spinner('saving ...'):
        trainX, trainy, fn = get_faces_array_label_filename(face_folder)
        numpy.savez_compressed(dataset, trainX, trainy, fn)
    return dataset


def get_faces_array_label_filename(directory):
    X, y, fn = list(), list(), list()

    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if not os.path.isdir(path):
            continue

        faces_array, file_names = get_faces_array_names(path)
        # create labels
        labels = [subdir for _ in range(len(faces_array))]
        st.write(labels)
        st.write(f"> Loaded {len(faces_array)} examples for class: {subdir}")
        # store
        X.extend(faces_array)
        y.extend(labels)
        fn.extend(file_names)
    return numpy.asarray(X), numpy.asarray(y), numpy.asarray(file_names)


def is_image_file(filename):
    return filename.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))


def get_faces_array_names(directory):
    face_array_and_name = []
    
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if is_image_file(path):
            face = extract_face_to_array(path)
            if face is not None:
                face_array_and_name.append((face, filename))
            st.info(f'loading: {path} {len(face_array_and_name)}')
    
    faces_array, file_names = zip(*face_array_and_name) if face_array_and_name else ([], [])
    return list(faces_array), list(file_names)


def extract_face_to_array(filename, downsize_ratio=0.7, required_size=(160, 160)):

    image = PIL.Image.open(filename)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    pixels = numpy.asarray(image)

    if pixels.shape[1] >= 1600 and pixels.shape[0] >= 1600:
        new_size = (int(pixels.shape[1] * downsize_ratio), int(pixels.shape[0] * downsize_ratio))
        pixels = numpy.array(PIL.Image.fromarray(pixels).resize(new_size))
    
    results = detector.detect_faces(pixels)     # extract the bounding box from the first face
    if len(results) == 0:
        return None             # No faces detected

    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]

    # resize pixels to the model size
    image = PIL.Image.fromarray(face)
    image = image.resize(required_size)
    face_array = numpy.asarray(image)
    return face_array
