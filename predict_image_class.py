# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 20:53:32 2022

@author: Grunk
"""

import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

st.header("Mushroom Image Classifier")
st.write("Using a MobileNetV3 model to classify the image.")

show_sample_image = st.sidebar.button("Show Sample Image")
generate_sample_prediction = st.sidebar.button("Predict Sample Image")

upload_file = st.sidebar.file_uploader("Upload Mushroom Images", type = ['jpg','png','jpeg'])
generate_prediction = st.sidebar.button("Predict Image")

model = tf.keras.models.load_model('MN_Full_model.h5', custom_objects ={'KerasLayer':hub.KerasLayer})
sample_image = Image.open('sample_image_rm.jpg')

if show_sample_image:
    with st.expander('image', expanded=True):
        st.image(sample_image, use_column_width=True)
        
class_names=['Amanita Bisporigera',
                 'Amanita Muscaria',
                 'Boletus Edulis',
                 'Cantharellus',
                 'Russula Mariae'
                 ]

def predict_image(upload_image):
    size=(224,224)
    image = ImageOps.fit(upload_image,size)
    image2 = tf.keras.applications.mobilenet_v3.preprocess_input(image)
    img = np.expand_dims(image2, axis=0)
    img_reshape=img.reshape((-1,224,224,3))
    prediction = model.predict(img_reshape).flatten()
    confidences={class_names[i]: float(prediction[i]) for i in range(5)}
    return confidences

def predict_sample_image(upload_image):
    size=(224,224)
    image = ImageOps.fit(sample_image,size)
    image2 = tf.keras.applications.mobilenet_v3.preprocess_input(image)
    img = np.expand_dims(image2, axis=0)
    img_reshape=img.reshape((-1,224,224,3))
    prediction = model.predict(img_reshape).flatten()
    confidences={class_names[i]: float(prediction[i]) for i in range(5)}
    return confidences  

if generate_prediction:
    image=Image.open(upload_file)
    with st.expander('image', expanded=True):
        st.image(image, use_column_width=True)
    predictions=predict_image(image)
    print(predictions)
    st.write(predictions)
    
if generate_sample_prediction:
    with st.expander('image', expanded=True):
        st.image(sample_image, use_column_width=True)
    image3=sample_image
    predictions=predict_sample_image(image3)
    print(predictions)
    st.write(predictions)