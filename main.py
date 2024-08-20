import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import cv2

# Load your pre-trained model
model = load_model("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/DEEP LEARNING/DL Project 2. MNIST Digit Classification with Neural Network/Proper/mnist_digit_classification_model.h5")

# Streamlit UI
st.title("Handwritten Digit Recognition")

# Create a canvas component with black background and white stroke
canvas_result = st_canvas(
    fill_color="#000000",  # Black fill color
    stroke_width=10,
    stroke_color="#ffffff",  # White stroke color
    background_color="#000000",  # Black background color
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button('Predict'):
    if canvas_result.image_data is not None:
        try:
            # Convert canvas image to grayscale
            input_image = canvas_result.image_data.astype(np.uint8)
            grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            
            # Resize the image to 28x28 pixels
            input_image_resize = cv2.resize(grayscale, (28,28))
            
            # Normalize the image
            input_image_resize = input_image_resize / 255.0
            
            # Reshape the image to match the input shape of the model
            image_reshaped = np.reshape(input_image_resize, [1, 28, 28, 1])
            
            # Debugging: Display the preprocessed image
            st.image(input_image_resize, caption="Resized and Normalized Image", use_column_width=False, clamp=True, channels='GRAY')
            
            # Predict the digit
            input_prediction = model.predict(image_reshaped)
            input_pred_label = np.argmax(input_prediction)
            
            # Debugging: Display the prediction probabilities
            st.write(f"Prediction Probabilities: {input_prediction}")
            
            # Display the prediction
            st.write(f"The Handwritten Digit is recognised as: {input_pred_label}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
