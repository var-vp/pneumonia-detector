import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("pneumonia_model.h5")

# App title and description
st.set_page_config(page_title="Pneumonia Detection App", layout="centered")
st.title("🩺 Pneumonia Detection from Chest X-rays")
st.markdown("""
Welcome to the *AI-powered Pneumonia Detector*! Upload a chest X-ray image, and the model will predict whether the patient has pneumonia or not.
""")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["png", "jpg", "jpeg"])

# Prediction function
def predict_pneumonia(image):
    _, img_height, img_width, channels = model.input_shape
    img = image.convert("L") if channels == 1 else image.convert("RGB")
    img = img.resize((img_width, img_height))
    img = np.array(img) / 255.0
    if channels == 1:
        img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    return "✅ Normal" if prediction > 0.5 else "🚨 Pneumonia Detected"

# Display uploaded image and make prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    if st.button("🧪 Analyze Image"):
        with st.spinner("Analyzing the X-ray..."):
            result = predict_pneumonia(image)
        st.success(f"Prediction: {result}")

# Footer
st.markdown("---")
