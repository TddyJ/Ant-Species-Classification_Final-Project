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
    model = tf.keras.models.load_model('revised_base_model_v2.h5')
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

    st.markdown("<div class='title-container'><h1>Ant Species Classification</h1></div>", unsafe_allow_html=True)
    st.markdown("---")

    # File Uploader
    file = st.file_uploader("Choose an Ant image among the following ant species: Fire Ant, Ghost Ant, Little Black Ant, Weaver Ant", type=["jpg", "png"])
    
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
            
            # Display the most likely class
            species = class_names[np.argmax(prediction)]
            st.success(f"OUTPUT : {species}")
            
            # Recommendation system
            recommendations = {
                'fire-ant': {
                    'description': 'Fire ants are known for their aggressive behavior and painful stings.',
                    'habitat': 'Typically found in open areas such as meadows and parks.',
                    'control_methods': 'Handle with caution. Can sting and cause irritation. Use insecticide specifically formulated for fire ants.'
                },
                'ghost-ant': {
                    'description': 'Ghost ants are tiny and often difficult to see due to their pale color.',
                    'habitat': 'Found in warm, humid environments, often indoors.',
                    'control_methods': 'Maintain clean environments and use baits containing boric acid to control ghost ants..'
                },
                'little-black-ant': {
                    'description': 'Little black ants are small and form large colonies.',
                    'habitat': 'Commonly found in wooded areas, but can also infest homes.',
                    'control_methods': 'Generally harmless but can infest food. Seal entry points and use ant baits to manage little black ants.'
                },
                'weaver-ant': {
                    'description': 'Weaver ants are known for building nests out of leaves.',
                    'habitat': 'Primarily found in tropical and subtropical regions.',
                    'control_methods': 'Can be beneficial for pest control in gardens. Avoid disturbing nests. Trim tree branches to reduce access points and use insecticidal soap to control weaver ants.'
                }
            }
            
            st.markdown("---")
            st.header("Species Information and Recommendations")
            st.write(f"**Description:** {recommendations[species]['description']}")
            st.write(f"**Habitat:** {recommendations[species]['habitat']}")
            st.write(f"**Control Methods:** {recommendations[species]['control_methods']}")
        except Exception as e:
            st.error("Error: Please upload an image file with one of the following formats: .JPG, .PNG, or .JPEG")


# Names Page
elif page == "Guide":
    st.title("Guide")
    st.markdown("---")
    st.write("This page displays the names of the classes that the model can classify:")
    st.markdown("---")
    st.write("- Fire Ant")
    fire_ant_image = load_image("https://drive.google.com/uc?export=download&id=1iCrDjIumbzrV5tH98IlRWYbgzIWtvF7J")
    st.image(fire_ant_image, use_column_width=True)
    st.markdown("---")
    st.write("- Ghost Ant")
    ghost_ant_image = load_image("https://drive.google.com/uc?export=download&id=1UiBZjPZd-vkULxpR6UFOVPzNUd0WIcMq")
    st.image(ghost_ant_image, use_column_width=True)
    st.markdown("---")
    st.write("- Little Black Ant")
    little_black_ant_image = load_image("https://drive.google.com/uc?export=download&id=1YotDPhmbaQasKqlc7lxt9pBp2H7DImWj")
    st.image(little_black_ant_image, use_column_width=True)
    st.markdown("---")
    st.write("- Weaver Ant")
    weaver_ant_image = load_image("https://drive.google.com/uc?export=download&id=1Y9dvo6M62qzze2B2RWnOgO-hrPrXAJQF")
    st.image(weaver_ant_image, use_column_width=True)
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
    st.write("[Click Here](https://github.com/TddyJ/Ant-Species-Classification_-Final-Project_-Group-8.git)")
    st.header("Google Drive Link")
    st.write("[Click Here](https://drive.google.com/drive/folders/14gjngXopMnuYPR4Ww366tT-fX3zd3JrD?usp=sharing)")
    st.header("Sample Images for Testing")
    st.write("[Click Here](https://drive.google.com/drive/folders/1nag4zj9Q8tpl2YZeYiTjaCoR4yvAWBvg?usp=sharing)")