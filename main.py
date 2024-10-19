import streamlit as st
import numpy as np
from PIL import Image
import keras

# List of disease labels
label_name = [
    'Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
    'Cherry healthy', 'Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight',
    'Corn healthy', 'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy',
    'Peach Bacterial spot', 'Peach healthy', 'Pepper bell Bacterial spot', 'Pepper bell healthy',
    'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch',
    'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight',
    'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites', 'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy'
]

# Streamlit description
st.write("""
The leaf disease detection model is built using deep learning techniques. It leverages transfer learning to recognize 
33 different types of leaf diseases. Please upload images of leaves from Apple, Cherry, Corn, Grape, Peach, Pepper, 
Potato, Strawberry, and Tomato for optimal results.
""")

# Load pre-trained model
model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')

# File uploader for leaf image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Open and process image using PIL
    img = Image.open(uploaded_file)

    # Preprocess the image (resize and normalize)
    resized_img = img.resize((150, 150))
    img_array = np.array(resized_img) / 255.0
    normalized_image = np.expand_dims(img_array, axis=0)

    # Show the uploaded image in Streamlit
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Make predictions
    predictions = model.predict(normalized_image)

    # Show the result if confidence is high
    confidence = predictions[0][np.argmax(predictions)] * 100
    if confidence >= 80:
        st.write(f"Result: {label_name[np.argmax(predictions)]} (Confidence: {confidence:.2f}%)")
    else:
        st.write("The model is not confident. Try uploading another image.")
