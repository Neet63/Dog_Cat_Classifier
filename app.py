import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model

# Load the saved Keras model
model = load_model('dog_cat_classifier.h5')

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    resized_image = cv2.resize(image, (224, 224))
    # Normalize pixel values to be in the range [0, 1]
    normalized_image = resized_image / 255.0
    # Expand dimensions to match the shape expected by the model
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image

# Function to make predictions
def predict(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Make prediction
    prediction = model.predict(preprocessed_image)
    # Return the predicted class
    return "Dog" if prediction[0][1] > prediction[0][0] else "Cat"

# Streamlit app
def main():
    st.title("Cat or Dog Classifier")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Predict
        if st.button('Predict'):
            img_array = np.array(image)
            result = predict(img_array)
            st.write('Prediction:', result)

if __name__ == '__main__':
    main()
