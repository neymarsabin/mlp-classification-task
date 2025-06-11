import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image

def main():
    print("Hello from ml-classification!")
    model = tf.keras.models.load_model("classification-model.h5")
    class_names = ['Bird', 'Car', 'Elephant', 'Flower', 'Human']

    img_height = 128
    img_width = 128

    st.set_page_config(page_title="Image Classifier", layout="centered")

    st.title("🧠 MLP Image Classifier")
    st.write("Upload an image of a **Bird**, **Car**, **Elephant**, **Flower**, or **Human**.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img = image.resize((img_width, img_height))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0  # Normalize and batch

        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index] * 100
        predicted_label = class_names[predicted_index]

        st.success(f"🧾 Prediction: **{predicted_label}** ({confidence:.2f}%)")


if __name__ == "__main__":
    main()
