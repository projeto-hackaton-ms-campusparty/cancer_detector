import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

def run():
    st.set_page_config(page_title="Cancer de mama Detector", page_icon="📈")

model = tf.keras.models.load_model("saved_model/IDC_model.h5")

st.markdown('<h1 style="text-align: center;">Identificar câncer de mama</h1>', unsafe_allow_html=True)
st.title('Identificação de IDC e Metastases por imagens, usando rede neural, e identificação por variáveis')



### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

if __name__ == '__main__':
    run()