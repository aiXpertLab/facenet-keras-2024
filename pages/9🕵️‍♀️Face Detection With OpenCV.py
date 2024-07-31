import streamlit as st,os
from utils import streamlit_components
from streamlit_extras.stateful_button import button

streamlit_components.streamlit_ui('🦣 Face Detection With OpenCV')
# -------------------------------------------------------------------------------------
from cv2 import (
    imread,
    imshow,
    waitKey,
    destroyAllWindows,
    CascadeClassifier,
    rectangle,
)
st.write(os.getenv('YCC'))

if button("Show Picture?", key="but1"):

    pixels = imread('./data/images/m2.jpg')# load the photograph
    classifier = CascadeClassifier('./models/haarcascade_frontalface_default.xml')# load the pre-trained model

    # perform face detection
    bboxes = classifier.detectMultiScale(pixels,1.1,3)      #scaleFactor and minNeighbors

    for box in bboxes:
        st.info(box)    # print bounding box for each detected face
    
        x, y, width, height = box   # extract
        x2, y2 = x + width, y + height
        
        rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 1) # draw a rectangle over the pixels

    imshow('face detection', pixels)    # show the image
    waitKey(6800)  # keep the window open until we press a key
    destroyAllWindows() # close the window
