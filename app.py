import streamlit as st
import tensorflow as tf
import numpy as np
import tempfile

from preprocessing.image_preprocessing import preprocess_image

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Hybrid Explainable Alzheimer's Disease Prediction",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# LOAD MODEL
# ======================================================
from models.hybrid_model import build_hybrid_model

@st.cache_resource
def load_model():

    model, _, _ = build_hybrid_model(
        input_shape=(224,224,3),
        num_classes=4
    )

    model.load_weights("final_hybrid_cbam_cleaned.weights.h5")

    return model

model = load_model()

# ======================================================
# CLASS LABELS
# ======================================================

class_names = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🧠 Project Information")

st.sidebar.markdown("""
### Hybrid Explainable Deep Learning Model

**Backbone Networks**
- ResNet50
- EfficientNetB3

**Attention Module**
- CBAM 

**Explainability**
- Grad-CAM 

**Input**
- Brain MRI Image

**Output**
- Alzheimer's Disease Stage
""")

# ======================================================
# MAIN PAGE
# ======================================================

st.title("🧠 Alzheimer's Disease Prediction System")

st.write("""
Upload a brain MRI image to predict the Alzheimer's disease stage using the trained Hybrid ResNet50 + EfficientNetB3 + CBAM model.
""")

st.markdown("---")

# ======================================================
# FILE UPLOADER
# ======================================================

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ======================================================
# PREDICTION
# ======================================================

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded MRI Image")

        st.image(
            uploaded_file,
            use_container_width=True
        )

        # Save uploaded image temporarily
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        )

        temp_file.write(uploaded_file.getbuffer())
        temp_file.close()

        temp_path = temp_file.name

    with col2:

        st.subheader("Prediction")

        if st.button("Predict"):

            with st.spinner("Analyzing MRI Image..."):

                image = preprocess_image(temp_path)

                if image is None:
                    st.error("Unable to preprocess the uploaded image.")
                    st.stop()

                image = np.expand_dims(image, axis=0)
                image = image.astype(np.float32)

                # Debug Information
                st.write("Image Shape:", image.shape)
                st.write("Image Dtype:", image.dtype)

                prediction = model.predict(image, verbose=0)

                probs = prediction[0]

                predicted_index = np.argmax(probs)

                confidence = probs[predicted_index] * 100

                st.success(
                    f"Predicted Stage: **{class_names[predicted_index]}**"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.markdown("---")

                st.subheader("Prediction Probabilities")

                for label, score in zip(class_names, probs):

                    st.write(f"**{label}**")

                    st.progress(float(score))

                    st.write(f"{score*100:.2f}%")

                st.markdown("---")

                st.info(
                        "Grad-CAM visualizations were generated separately for model interpretability and are presented in the Results section."
                )

st.markdown("---")

st.caption(
    "Mini Project | M.Tech Artificial Intelligence | Alzheimer's Disease Prediction using MRI"
)
