# If you want a more interactive Windows app with a web UI, use Streamlit:

# Install Streamlit: pip install streamlit
# Create a script (streamlit_app_face.py):


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("incremental_vgg16_fer2013.h5")
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def predict_emotion(img_path):
    img = image.load_img(img_path, target_size=(48, 48))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return emotion_labels[predicted_class]

st.title("Facial Emotion Recognition")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    emotion = predict_emotion(uploaded_file)
    st.write(f"Predicted Emotion: {emotion}")
# Run the app:streamlit run app.py
# This will open a browser-based Windows app.