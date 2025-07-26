import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2


# Load Model
@st.cache_resource
def load_change_model():
    return load_model("main_model_h5.h5")

model = load_change_model()


st.title("🛰️ Change Detection with Satellite Images")

# Upload Images 
uploaded_img1 = st.file_uploader("Timezone 1 Image (T1)", type=["png", "jpg", "jpeg"])
uploaded_img2 = st.file_uploader("Timezone 2 Image (T2)", type=["png", "jpg", "jpeg"])

if uploaded_img1 and uploaded_img2:
    # Show images
    img1 = Image.open(uploaded_img1).convert("RGB")
    img2 = Image.open(uploaded_img2).convert("RGB")

    st.subheader("Loaded Images")
    st.image([img1, img2], caption=["Image T1", "Image T2"], width=250)

    # Resize and normalize ımages
    IMG_SIZE = 256  
    def preprocess(img):
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img) / 255.0
        return img

    img1 = preprocess(img1)
    img2 = preprocess(img2)

    
    input_image = np.concatenate([img1, img2], axis=-1)  # (H, W, 6)
    input_image = np.expand_dims(input_image, axis=0)    # (1, H, W, 6)

    # Predict
    prediction = model.predict(input_image)[0, :, :, 0]

    # Show result
    st.subheader("Change Detection (Mask)")
    st.image(prediction, caption="Change Mask", use_container_width=True, clamp=True,width=300)
