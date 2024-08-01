import os
import numpy

import streamlit as st

from sklearn import preprocessing
from sklearn.svm import SVC

from utils.face_processing import load_faces_from_train_val_prod, load_faces_prod


def save_dataset(datasize_name, train, val):
    with st.spinner('saving ...'):
        trainX, trainy = load_faces_from_train_val_prod(train)
        testX,  testy  = load_faces_from_train_val_prod(val)
        # metadata = {
        #     'train_dir': train,
        #     'val_dir': val,
        #     'num_train_samples': len(trainX),
        #     'num_val_samples': len(testX)
        # }
        # save arrays to one file in compressed format
        numpy.savez_compressed(datasize_name, trainX, trainy, testX, testy)
    return datasize_name


def load_dataset_with_metadata(dataset_name):
    with numpy.load(dataset_name, allow_pickle=True) as data:
        trainX = data['trainX']
        trainy = data['trainy']
        testX = data['testX']
        testy = data['testy']
        # metadata = data['metadata'].item()  # .item() to convert array to dictionary
    return trainX, trainy, testX, testy


def save_dataset_prod(dataset, face_folder):
    # save one set of dataset
    
    with st.spinner('saving ...'):
        trainX, trainy = load_faces_prod(face_folder)
        # metadata = {
        #     'train_dir': train,
        #     'val_dir': val,
        #     'num_train_samples': len(trainX),
        #     'num_val_samples': len(testX)
        # }
        # save arrays to one file in compressed format
        numpy.savez_compressed(dataset, trainX, trainy)
    return dataset

