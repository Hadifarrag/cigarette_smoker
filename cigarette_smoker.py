import streamlit as st 
import tensorflow as tf
import cv2 as cv
import numpy as np
from PIL import Image
print(tf.__version__)
def load_model_from_hdf5(file_path):
    try:
        # Load model with custom objects and compile=False
        model = tf.keras.models.load_model(
            file_path,
            compile=False
        )
        
        # Compile the model after loading
       # model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image,target_size=(350,350)):
    try:
        # Convert PIL image to numpy array
        image_array = np.array(image)
        
        # Convert RGB to BGR if necessary (OpenCV uses BGR)
       # if len(image_array.shape) == 3 and image_array.shape[2] == 3:
       #     image_array = cv.cvtColor(image_array, cv.COLOR_RGB2BGR)
        
        # Resize image to target size
        image_resized = cv.resize(image_array, target_size)
        
        # Normalize pixel values to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Expand dimensions for batch size
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

class_labels = {"Not Smoking": 0, "Smoking": 1}

st.title("Cigarette Detection APP")
st.header("Upload photo and check if there are smokers in it")

# Upload the image
uploaded_file = st.file_uploader("Upload the image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open the uploaded file as a PIL image
        image_input = Image.open(uploaded_file)
        
        # Display the image using Streamlit
        st.image(image_input, caption="Uploaded Image", use_column_width=True)

        if st.button("Check image"):
            # Preprocess the image
            processed_image = preprocess_image(image_input)
            
            if processed_image is not None:
                # Load and compile the model
                model = load_model_from_hdf5('F:\\github\\streamlit\\cigarette_smoker\\cigarette_smoker.hdf5')
                
                if model is not None:
                    # Make prediction
                    prediction = model.predict(processed_image)
                    predicted_class_label = np.argmax(prediction)
                    result = list(class_labels.keys())[predicted_class_label]
                    
                    # Display result
                    st.write(f"Prediction: {result}")
                    st.write(f"Confidence: {prediction[0][predicted_class_label]:.2%}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")