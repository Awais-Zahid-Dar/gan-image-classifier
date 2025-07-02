import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import base64

# -------------------------------
# Page Config & Background Setup
# -------------------------------
st.set_page_config(page_title="GAN Image Classifier", layout="centered")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .title {{
        color: white;
        text-shadow: 2px 2px 4px #000;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

add_bg_from_local('background.jpg')

# -------------------------------
# Load Model (Cached)
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/mobilenet_fakeness_classifier.h5")

model = load_model()
class_names = ['DCGAN', 'Real', 'StackGAN', 'StyleGAN']

# -------------------------------
# Preprocess Image
# -------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------
# UI Layout
# -------------------------------
st.markdown("<h1 class='title' style='text-align:center;'>🧠 GAN Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='title' style='text-align:center;'>Detect Real vs GAN-Generated Images (DCGAN, StackGAN, StyleGAN)</h4>", unsafe_allow_html=True)
st.markdown("##")

uploaded_file = st.file_uploader("📤 Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.markdown("### ⏳ Classifying...")

    with st.spinner("Running inference..."):
        processed = preprocess_image(image)
        predictions = model.predict(processed)
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = predictions[0][predicted_index] * 100

    st.success(f"🎯 **Prediction: {predicted_class}** with **{confidence:.2f}% confidence**")
    st.markdown("### 🔍 Prediction Breakdown")
    st.bar_chart(predictions[0])

else:
    st.warning("👆 Please upload an image to classify.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("<center><span style='color:white;'>Created with ❤️ using Streamlit</span></center>", unsafe_allow_html=True)
