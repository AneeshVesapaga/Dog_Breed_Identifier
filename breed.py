import streamlit as st
import numpy as np
from PIL import Image
import joblib


# Attempt to import cv2 and handle potential import errors
try:
    import cv2
    st.success('cv2 imported successfully.')
except ImportError as e:
    st.error(f'Error importing cv2: {e}')

# Display title
st.title('Dog Breed Classification App')

# Display an example image
image_path = 'ino_img.jpg'  # Replace with your actual image file path
st.image(image_path, caption='Example Image')

# Load the pre-trained model
model_path = r"lr.pkl"
try:
    model = joblib.load(model_path)
    st.success('Model loaded successfully.')
except Exception as e:
    st.error(f'Error loading model: {e}')

# Function to preprocess the image
def preprocess_image(image):
    try:
        # Resize the image to (200, 200) as per your model's input requirement
        resized_image = image.resize((200, 200))
        # Convert to numpy array and flatten
        img_array = np.array(resized_image)
        # Check if the image has three color channels (RGB), if so, convert it to grayscale
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        flattened_img = img_array.flatten()
        # Ensure the flattened image has exactly 40000 features
        if flattened_img.shape[0] != 40000:
            raise ValueError(f"Expected 40000 features, but got {flattened_img.shape[0]} features.")
        # Reshape to (1, 40000) to match model's input shape
        processed_image = flattened_img.reshape(1, -1)
        return processed_image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Streamlit app
def main():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])  # Adjust type as per your image types

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            if st.button('Submit'):
                # Preprocess the image
                processed_image = preprocess_image(image)

                if processed_image is not None:
                    # Make prediction
                    prediction = model.predict(processed_image)
                    st.write(f'Raw Prediction: {prediction}')
                    
                    # Optionally, map numerical prediction to labels if needed
                    # predicted_class = 'cat' if prediction == 0 else 'dog'
                    # st.write(f'Predicted Class: {predicted_class}')
        
        except Exception as e:
            st.error(f"Error processing or classifying image: {e}")

if __name__ == '__main__':
    main()


