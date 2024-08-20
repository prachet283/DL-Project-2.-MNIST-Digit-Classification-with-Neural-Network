# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 18:17:20 2024

@author: prachet
"""

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your pre-trained model
model = load_model("mnist_digit_classification_model.h5")

# Streamlit UI
st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, 1)
    
    # Display the uploaded image
    st.image(input_image, channels="BGR", caption="Uploaded Image")
    
    # Convert to grayscale
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to 28x28 pixels
    input_image_resize = cv2.resize(grayscale, (28,28))
    
    # Normalize the image
    input_image_resize = input_image_resize / 255.0
    
    # Reshape the image to match the input shape of the model
    image_reshaped = np.reshape(input_image_resize, [1, 28, 28, 1])
    
    # Predict the digit
    input_prediction = model.predict(image_reshaped)
    input_pred_label = np.argmax(input_prediction)
    
    # Display the prediction
    st.write(f"The Handwritten Digit is recognised as: {input_pred_label}")
