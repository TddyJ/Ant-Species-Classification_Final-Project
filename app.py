import os
import streamlit as st
import tensorflow as tf
import PIL
from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('base_model.h5')
    return model

@st.cache
def load_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    return image
    
model = load_model()

# Navigation Bar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Guide", "About", "Links"])

# Home Page
if page == "Home":
    st.markdown(
    """
    <style>
    /* Add a border around the title */
    .title-container {
        border: 2px solid #f63366;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.markdown("<div class='title-container'><h1>Brain Tumor MRI Classification</h1></div>", unsafe_allow_html=True)
    st.markdown("---")

    # File Uploader
    file = st.file_uploader("Choose an image of an ant among the following species:\n Fire Ant \n Ghost Ant \n Little Black Ant \n Weaver Ant", type=["jpg", "png"])
    
    # Function to make predictions
    def import_and_predict(image_data, model):
        size = (150, 150)  
        image = ImageOps.fit(image_data, size, PIL.Image.LANCZOS) 
        img = np.asarray(image)
        img = img / 255.0  
        img_reshape = img[np.newaxis, ...]
        prediction = model.predict(img_reshape)
        return prediction
    
    if file is None:
        st.text("Please upload an image file")
    else:
        try:
            image = Image.open(file)
            st.image(image, use_column_width=True, output_format='JPEG')
            
            # Add a border to the image
            st.markdown(
                "<style> img { display: block; margin-left: auto; margin-right: auto; border: 2px solid #ccc; border-radius: 8px; } </style>",
                unsafe_allow_html=True
            )
            
            prediction = import_and_predict(image, model)
            class_names = ['fire-ant', 'ghost-ant', 'little-black-ant', 'weaver-ant']
            
            # Display confidence levels
            confidence_levels = {class_name: round(float(pred), 4) for class_name, pred in zip(class_names, prediction[0])}
            st.write("Confidence levels:")
            for class_name, confidence in confidence_levels.items():
                st.write(f"{class_name}: {confidence * 100:.2f}%")
            
            # Display the most likely class
            string = "OUTPUT : " + class_names[np.argmax(prediction)]
            st.success(string)
        except Exception as e:
            st.error("Error: Please upload an image file with one of the following formats: .JPG, .PNG, or .JPEG")


# Names Page
elif page == "Guide":
    st.title("Guide")
    st.markdown("---")
    st.write("This page displays the names of the classes that the model can classify:")
    st.markdown("---")
    st.write("- Fire Ant")
    glioma_image = load_image("https://drive.google.com/uc?export=view&id=1_dHlhzdvtZxzPKiby1w9N__R9uPrAXUP")
    st.image(glioma_image, use_column_width=True)
    st.markdown("---")
    st.write("- Ghost Ant")
    meningioma_image = load_image("https://drive.google.com/uc?export=view&id=1gCTR9Oe4zuE3SDojoqYPMPwOupfSA9Lf")
    st.image(meningioma_image, use_column_width=True)
    st.markdown("---")
    st.write("- Little Black Ant")
    no_tumor_image = load_image("https://drive.google.com/uc?export=view&id=1JqI8bUEW6P3PyYfGsudr_0oMxekgYLDy")
    st.image(no_tumor_image, use_column_width=True)
    st.markdown("---")
    st.write("- Weaver Ant")
    pituitary_image = load_image("https://drive.google.com/uc?export=view&id=1gLzYhPu_P-ZZybapSBEE_mzTymFCd7FP")
    st.image(pituitary_image, use_column_width=True)
    st.markdown("---")
    
# About Page
elif page == "About":
    st.title("About")
    st.markdown("---")
    st.write("This is a simple web application that classifies Ant images among the following species: Fire Ant, Ghost Ant, Little Black Ant, and Weaver Ant")
    st.write("It uses a deep learning model trained on different Ant images to make predictions.")
    st.markdown("---")
    st.header("Group 8 - CPE 313-CPE32S8")
    st.markdown("---")
    st.write("Rojo, Maverick")
    st.write("Roque, Jared Miguel")
    st.markdown("---")

elif page == "Links":
    st.title("Links")
    st.markdown("---")
    st.header("Github Link")
    st.write("[Click Here](https://github.com/qmjae/Brain-Tumor-MRI-Classification-using-Streamlit)")
    st.header("Google Drive Link")
    st.write("[Click Here](https://drive.google.com/drive/folders/1MExGDFt6MVJunB97RloUM7sNb3rudecz?usp=sharing)")
    st.header("Sample Images for Testing")
    st.write("[Click Here](https://drive.google.com/drive/folders/1gL6A_zjZQDYCsw8UpP-wvto6HKScBMuk?usp=drive_link)")