import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import cv2

model = load_model("mnist_digit_classification_model.h5")

st.title("Handwritten Digit Recognition")

canvas_result = st_canvas(
    fill_color="#000000", 
    stroke_width=10,
    stroke_color="#ffffff",  
    background_color="#000000", 
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button('Predict'):
    if canvas_result.image_data is not None:
        try:
            input_image = canvas_result.image_data.astype(np.uint8)
            grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            
            input_image_resize = cv2.resize(grayscale, (28,28))
            
            input_image_resize = input_image_resize / 255.0
            
            image_reshaped = np.reshape(input_image_resize, [1, 28, 28, 1])
            
            st.image(input_image_resize, caption="Resized and Normalized Image", use_column_width=False, clamp=True, channels='GRAY')
            
            input_prediction = model.predict(image_reshaped)
            input_pred_label = np.argmax(input_prediction)
            
            st.write(f"Prediction Probabilities: {input_prediction}")
            
            # Display the prediction
            st.write(f"The Handwritten Digit is recognised as: {input_pred_label}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
