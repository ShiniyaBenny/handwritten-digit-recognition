import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load your trained model (replace 'model.h5' with your actual model file)
model = load_model('mnist_cnn_model.h5')


st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image to match model input
    image = image.resize((28, 28))
    image = ImageOps.invert(image)
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image)
    digit = np.argmax(prediction)

    st.write(f"Predicted Digit: **{digit}**")
