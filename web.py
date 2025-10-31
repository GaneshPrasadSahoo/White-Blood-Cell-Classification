import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# ✅ Load the trained model
model_path = r"C:\Users\Ganesh prasad sahoo\OneDrive\Documents\Downloads\All DataSet\White_blood_cells_classification\cnn_model.h5"
model = tf.keras.models.load_model(model_path)

# ✅ Define class labels
class_labels = ["Neutrophil", "Eosinophil", "Basophil", "Lymphocyte", "Monocyte"]

# ✅ Function to preprocess the image
def preprocess_image(image):
    img = np.array(image, dtype=np.uint8)  # Convert PIL image to NumPy array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR if needed

    # ✅ Resize image to match model input shape
    target_size = model.input_shape[1:3]
    img = cv2.resize(img, target_size)

    img = img.astype(np.float32) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# ✅ Function to predict white blood cell type
def predict_wbc(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]

# 🌟 Streamlit UI
st.title("🧪 White Blood Cell Classification")
st.write("Upload an image of a white blood cell to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')  # Ensure image is in RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)  # Display image
    
    # ✅ Make a prediction
    prediction = predict_wbc(image)
    
    # ✅ Display prediction result
    st.write(f"**🧪 Predicted White Blood Cell Type:** {prediction}")
