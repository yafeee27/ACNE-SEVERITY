import streamlit as st
import numpy as np
import os

from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# =====================
# Load Model
# =====================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "final_model.keras"
)


@st.cache_resource
def load_acne_model():
    """Load the trained model with caching for performance."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Model file not found: {MODEL_PATH}\n\n"
            "Jalankan `python train.py` terlebih dahulu untuk melatih model."
        )
        st.stop()
    return load_model(MODEL_PATH)


model = load_acne_model()

# =====================
# Class Names
# =====================

class_names = [
    "Parah",
    "Ringan",
    "Sedang"
]

# =====================
# Recommendation
# =====================

recommendation = {

    "Ringan":
    """
    ✅ Cuci wajah 2x sehari

    ✅ Gunakan moisturizer

    ✅ Gunakan sunscreen
    """,

    "Sedang":
    """
    ✅ Salicylic Acid

    ✅ Benzoyl Peroxide

    ✅ Menjaga kebersihan wajah
    """,

    "Parah":
    """
    ⚠️ Disarankan konsultasi ke dokter kulit
    """
}

# =====================
# UI
# =====================

st.title(
    "Klasifikasi Tingkat Keparahan Jerawat"
)

st.write(
    "Upload foto wajah atau ambil foto langsung untuk mengetahui tingkat keparahan jerawat."
)

tab_camera, tab_upload = st.tabs(["📷 Ambil Foto", "📁 Upload Gambar"])

with tab_camera:
    camera_file = st.camera_input("Ambil foto wajah")

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload Gambar",
        type=["jpg", "jpeg", "png"]
    )

# Pilih input: kamera atau upload
input_file = camera_file or uploaded_file

# =====================
# Prediction
# =====================

if input_file is not None:

    img = Image.open(input_file)

    # Convert RGBA to RGB (for PNG with transparency)
    if img.mode == "RGBA":
        img = img.convert("RGB")

    st.image(
        img,
        caption="Gambar Upload",
        use_container_width=True
    )

    img = img.resize((224, 224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )

    idx = np.argmax(prediction)

    confidence = np.max(prediction)

    severity = class_names[idx]

    st.subheader(
        f"Hasil Prediksi: {severity}"
    )

    st.write(
        f"Confidence Score: {confidence*100:.2f}%"
    )

    st.subheader(
        "Rekomendasi Perawatan"
    )

    st.write(
        recommendation[severity]
    )