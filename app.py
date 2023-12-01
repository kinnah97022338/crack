import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model_path = 'concrete_crack_model.h5'
model = load_model(model_path)

# Define the target size for images
target_size = (224, 224)

# Preprocess the image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def main():
    st.title("Concrete Crack Detection")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the uploaded image for prediction
        processed_image = preprocess_image(uploaded_file)

        # Make a prediction
        prediction = model.predict(processed_image)

        # Display the prediction result
        if prediction[0][0] > 0.5:
            st.success("Crack Detected!")
        else:
            st.success("No Crack Detected!")

if __name__ == "__main__":
    main()