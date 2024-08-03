import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Define class mapping
class_mapping = {
    0: 'Normal',
    1: 'Malignant',
    2: 'Benign'
}

# Function to load and preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to match model input size
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to load and run inference on the model
def run_inference(image_array, model_path):
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Resize input tensor to accommodate the image
    interpreter.resize_tensor_input(input_details[0]['index'], [image_array.shape[0], 150, 150, 3])
    interpreter.allocate_tensors()
    
    interpreter.set_tensor(input_details[0]['index'], image_array.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# Define Streamlit app
def main():
    # Set up the page with icon and layout
    st.set_page_config(
        page_title="Breast Cancer Classification",
        page_icon="image/page_icon.png",  # Path to your icon image file
        layout='wide',
        initial_sidebar_state='expanded'
    )
    st.sidebar.markdown("# aibytec")
    st.sidebar.image('image/logo.jpg', width=200)
    st.title("Breast Cancer Classification using Ultrasound Images")

    # Custom CSS for styling
    st.markdown("""
    <style>
    body {
        animation: gradientAnimation 15s ease infinite;
        background-size: 400% 400%;
        background-image: linear-gradient(45deg, #EE7752, #E73C7E, #23A6D5, #23D5AB);
    }

    @keyframes gradientAnimation {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }

    .dataset-info {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        color: #333333;
    }

    .instructions {
        background-color: rgba(255, 255, 255, 0.6);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        color: #333333;
    }

    .caution-note {
        background-color: rgba(255, 255, 255, 0.4);
        padding: 15px;
        margin-bottom: 15px;
        border-left: 6px solid #FF5733;
        border-radius: 5px;
        box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.1);
        color: #333333;
    }

    .caution-note:last-child {
        margin-bottom: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Instructions
    st.markdown("""
    ## Instructions
    
    1. Upload an image file (JPEG or PNG format) of a breast ultrasound scan.
    2. Click the "Predict" button to get the Prediction result.
    """)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Create a two-column layout
        col1, col2 = st.columns([2, 1])  # Adjust the width ratio as needed

        with col1:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=350)
        
        with col2:
            # Display the prediction button and results
            if st.button("Predict"):
                try:
                    model_path = 'model.tflite'
                    image_array = preprocess_image(image)
                    output = run_inference(image_array, model_path)

                    # Get predicted class index with the highest score
                    predicted_class_index = np.argmax(output)
                    
                    # Get predicted class from class_mapping
                    predicted_class = class_mapping.get(predicted_class_index, "Unknown")
                    
                    # Display predicted class
                    st.write("Predicted class:", predicted_class)
                    
                    # Display cautionary note as a popup message
                    st.warning("""
                    **Accuracy Disclaimer**: The predictions made by this app are based on a machine learning model and may not always be 100% accurate. Use the results as a supplementary tool and consult medical professionals for definitive diagnosis.
                    """)

                except ValueError as e:
                    st.error(str(e))

    # Dataset Information
    st.markdown("""
    ## Dataset Information
    
    Breast cancer is one of the most common causes of death among women worldwide. Early detection helps in reducing the number of early deaths. The data reviews the medical images of breast cancer using ultrasound scans. Breast Ultrasound Dataset is categorized into three classes: normal, benign, and malignant images. Breast ultrasound images can produce great results in classification, detection, and segmentation of breast cancer when combined with machine learning.

    **Data**:
    The data collected at baseline include breast ultrasound images among women aged between 25 and 75 years old. This data was collected in 2018. The number of patients is 600 female patients. The dataset consists of 780 images with an average image size of 500x500 pixels. The images are in PNG format. The ground truth images are presented with original images. The images are categorized into three classes: normal, benign, and malignant.
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
