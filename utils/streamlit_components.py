import streamlit as st, os


def streamlit_ui(main_title):
    st.set_page_config(page_title='AI SQL Linguist 👋', page_icon="💯", ),
    st.title(main_title)  # not accepting default

    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://omnidevx.netlify.app/logo/aipro.png);
                background-size: 300px; /* Set the width and height of the image */
                background-repeat: no-repeat;
                padding-top: 80px;
                background-position: 15px 10px;
            }
        </style>
        """,
                unsafe_allow_html=True,
                )


def general():
    st.write(os.getenv('YCC'))
    st.markdown('''
        OpenCV provides the CascadeClassifier class that can be used to create a `cascade classifier` for face detection. 
                
        The constructor can take a filename as an argument that specifies the XML file for a pre-trained model. 
        
        OpenCV provides a number of pre-trained models as part of the installation. These are available on your system and are also available on the
        OpenCV GitHub project. 
                
        Download a pre-trained model for frontal face detection from the
        OpenCV GitHub project and place it in your current working directory with the filename
        `haarcascade frontalface default.xml`.
    '''
                )


def MTCNN():
    st.write('''
         
        #### Landmark Detection 
A number of deep learning methods have been developed and demonstrated for face detection. Perhaps one of the more popular approaches is called the Multi-Task Cascaded Convolutional
        Neural Network, or `MTCNN` for short. 
        
The network uses a cascade structure with three networks; 
- first the image is rescaled to a range of different sizes (called an image pyramid), then the first model (Proposal Network or
`P-Net`) proposes candidate facial regions, 
- the second model (Refine Network or `R-Net`) filters the bounding boxes, and 
- the third model (Output Network or `O-Net`) proposes facial landmarks. The proposed CNNs consist of three stages. 

In the first stage, it produces candidate windows quickly through a shallow CNN. Then, it refines the windows to reject a
large number of non-faces windows through a more complex CNN. Finally, it uses a more powerful CNN to refine the result and output facial landmarks positions.
         ''')
