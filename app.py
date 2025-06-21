import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import json
from PIL import Image

# Carregar modelo e classes
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cifar10_model.h5")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

# Função para classificação
def classify_image(image: Image.Image):
    img_array = np.array(image.convert("RGB"))
    img_resized = cv2.resize(img_array, (32, 32)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_input)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_names[class_id], confidence

# Interface
st.title("📸 Classificador de imagens")
st.write("Você pode **tirar uma foto com a webcam** ou **enviar uma imagem** para ser classificada.")

# Opções de entrada
tab1, tab2 = st.tabs(["📷 Usar Webcam", "📁 Upload de Imagem"])

# TAB 1 - Webcam
with tab1:
    cam_image = st.camera_input("Capture uma imagem com a webcam")
    if cam_image is not None:
        image = Image.open(cam_image)

        label, confidence = classify_image(image)
        st.success(f"Classe prevista: **{label}**")
        st.info(f"Confiança: **{confidence * 100:.2f}%**")

# TAB 2 - Upload
with tab2:
    uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem enviada", use_column_width=True)

        label, confidence = classify_image(image)
        st.success(f"Classe prevista: **{label}**")
        st.info(f"Confiança: **{confidence * 100:.2f}%**")
