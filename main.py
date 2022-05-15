import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
model = tf.keras.models.load_model('effnet.h5')


st.write("""
         # Brain Tumor Classification
         """
         )
st.write("This is a simple image classification web app to predict if there is a tumor")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


def import_and_predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    p = model.predict(img)

    return p


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    p = np.argmax(prediction, axis=1)[0]
    if p == 0:
        p = 'There is a Glioma Tumor with %' + str(100 * prediction[0][0])[0:4] + " percent"
    elif p == 1:
        p = 'There is a Meningioma Tumor with %' + str(100 * prediction[0][1])[0:4] + " percent"
    elif p == 2:
        p = 'There is a No Tumor with %' + str(100 * prediction[0][2])[0:4] + " percent"
        print('The model predicts that there is no tumor')
    else:
        p = 'There is a Pituitary Tumor with %' + str(100 * prediction[0][3])[0:4] + " percent"

    if p != 1:
        print(f'The Model predicts that it is a {p}')

    st.text("Probability (0: Glioma Tumor, 1: Meningioma Tumor, 2: No Tumor , 3:Pituitary Tumor")
    st.write(prediction)
    st.write(p)


