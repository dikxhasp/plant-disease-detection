import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="AI Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

@st.cache_resource
def load_ai_model():
    return load_model("model.keras")

model = load_ai_model()

with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

disease_info = {
    "Pepper__bell___Bacterial_spot":"Remove infected leaves and avoid overhead watering.",
    "Pepper__bell___healthy":"Healthy plant. No disease detected.",
    "Potato___Early_blight":"Use fungicide and remove infected leaves.",
    "Potato___Late_blight":"Avoid excess moisture and use recommended fungicide.",
    "Potato___healthy":"Healthy plant. No disease detected.",
    "Tomato___Early_blight":"Remove infected leaves and improve air circulation.",
    "Tomato___Late_blight":"Use fungicide and avoid wet leaves.",
    "Tomato___Leaf_Mold":"Reduce humidity and increase ventilation.",
    "Tomato___healthy":"Healthy plant. No disease detected."
}

st.title("🌿 AI Based Plant Disease Detection System")

st.markdown("""
Upload a leaf image and the trained CNN model will predict the disease.
""")

uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="📷 Uploaded Leaf", use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("🔍 Analyzing leaf image..."):
        prediction = model.predict(img, verbose=0)

    confidence = float(np.max(prediction) * 100)
    predicted_class = class_names[np.argmax(prediction)]

    top3_indices = np.argsort(prediction[0])[-3:][::-1]

    with col2:
        st.success(f" **Disease** **Prediction:** {predicted_class}")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)

        if "healthy" in predicted_class.lower():
            st.success("✅ This plant appears to be Healthy.")
        else:
            st.warning("⚠️ Disease detected. Check the recommendation below.")

        st.subheader("💡 Recommendation")

        if predicted_class in disease_info:
            st.info(disease_info[predicted_class])
        else:
            st.info("Disease detected plant is unhealthy. Please consult an expert for further guidance.")

    st.markdown("---")

    st.subheader("🏆 Top 3 Predictions")

    for index in top3_indices:
        st.write(
            f"**{class_names[index]}** : {prediction[0][index] * 100:.2f}%"
        )

st.markdown("---")

st.subheader("📌 About This Project")

st.write("""
This application uses a **Convolutional Neural Network (CNN)** trained on the
**PlantVillage Dataset** to identify diseases from plant leaf images.

### Features
- 🌿 Plant Disease Detection
- 🤖 CNN Deep Learning Model
- 📷 Image Upload
- 📊 Confidence Score
- 🏆 Top 3 Predictions
- 💡 Disease Recommendation

### Technologies Used
- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow

Developed as an **AI & Machine Learning College Project**.
""")

st.markdown("---")
st.caption("🌱 Developed using TensorFlow • CNN • Streamlit")
