import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load your trained model (provide the correct path to your model)
model = load_model("whiteblood.h5")

# Define labels
labels = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]

# Streamlit app
st.title("White Blood Cell Classification")

# Allow user to upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Preprocess the image (resize and reshape)
    img_resized = cv2.resize(img, (150, 150))  
    img_array = np.array(img_resized).reshape(1, 150, 150, 3)  # Reshape to (1, 150, 150, 3)

    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Predict the label using the trained model
    predictions = model.predict(img_array)
    predicted_label = labels[np.argmax(predictions)]

    # Display the predicted label
    st.write(f"Predicted Label: {predicted_label}")
