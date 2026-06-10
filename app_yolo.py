import streamlit as st
import numpy as np
import os
import cv2
from PIL import Image
from ultralytics import YOLO

# =====================
# Load Model
# =====================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "final_model_yolo.pt"
)


@st.cache_resource
def load_acne_model():
    """Load the trained YOLO model with caching for performance."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Model file YOLO tidak ditemukan: {MODEL_PATH}\n\n"
            "Jalankan `python train_yolo.py` terlebih dahulu untuk melatih model YOLO."
        )
        st.stop()
    return YOLO(MODEL_PATH)


# Load the model
try:
    model = load_acne_model()
except Exception as e:
    st.error(f"Gagal memuat model YOLO: {e}")
    st.stop()

# =====================
# Recommendation
# =====================

recommendation = {
    "Ringan":
    """
    ✅ Cuci wajah 2x sehari dengan pembersih lembut (gentle cleanser).
    
    ✅ Gunakan pelembap (moisturizer) non-comedogenic.
    
    ✅ Selalu gunakan tabir surya (sunscreen) di siang hari.
    """,

    "Sedang":
    """
    ✅ Gunakan produk dengan kandungan Salicylic Acid atau Benzoyl Peroxide.
    
    ✅ Jaga kebersihan wajah dan hindari menyentuh wajah dengan tangan kotor.
    
    ✅ Tetap gunakan pelembap dan sunscreen yang cocok untuk kulit berjerawat.
    """,

    "Parah":
    """
    ⚠️ Tingkat keparahan jerawat Anda cukup tinggi.
    
    ⚠️ Sangat disarankan untuk berkonsultasi langsung dengan dokter spesialis kulit (dermatolog) untuk mendapatkan pengobatan medis yang tepat (seperti retinoid topikal, antibiotik, atau terapi khusus).
    """
}

# =====================
# UI
# =====================

st.set_page_config(
    page_title="Klasifikasi Keparahan Jerawat (YOLO)",
    page_icon="📷",
    layout="centered"
)

st.title("Klasifikasi Tingkat Keparahan Jerawat (YOLOv8)")
st.write(
    "Unggah foto wajah atau ambil foto langsung untuk mengetahui tingkat keparahan jerawat menggunakan model YOLO."
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
    try:
        img = Image.open(input_file)

        # Convert RGBA to RGB (for PNG with transparency)
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # --- Face Detection ---
        # Convert PIL to BGR OpenCV format
        opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Load Haar Cascade Face Detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Grayscale conversion
        gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        # Check if no face detected
        if len(faces) == 0:
            st.image(
                img,
                caption="Gambar Input",
                use_container_width=True
            )
            st.error("❌ Wajah tidak terdeteksi!")
            st.warning("Mohon ambil atau unggah foto wajah Anda yang terlihat jelas, menghadap ke depan, dan memiliki pencahayaan yang cukup agar sistem dapat menilai tingkat keparahan jerawat.")
            st.stop()
            
        # Draw bounding boxes on faces
        img_for_display = opencv_img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_for_display, (x, y), (x+w, y+h), (0, 255, 0), 4)
            
        # Convert back to RGB for display
        img_with_box = Image.fromarray(cv2.cvtColor(img_for_display, cv2.COLOR_BGR2RGB))
        
        st.image(
            img_with_box,
            caption="Wajah Berhasil Terdeteksi (Kotak Hijau)",
            use_container_width=True
        )

        with st.spinner("Menganalisis tingkat keparahan jerawat menggunakan YOLOv8..."):
            # Jalankan prediksi (YOLO secara otomatis menangani resize ke 224x224 secara internal)
            results = model(img, verbose=False)
            
            # Mendapatkan hasil probabilitas
            probs = results[0].probs
            
            # Indeks kelas terbaik dan confidence score
            idx = probs.top1
            confidence = probs.top1conf.item()
            
            # Mengambil nama kelas (Parah / Ringan / Sedang)
            severity = results[0].names[idx]

        # Menampilkan Hasil
        st.success("Analisis Selesai!")
        
        # Berikan warna indikator berdasarkan kelas
        if severity == "Ringan":
            st.subheader(f"Hasil Prediksi: 🟢 {severity}")
        elif severity == "Sedang":
            st.subheader(f"Hasil Prediksi: 🟡 {severity}")
        else:
            st.subheader(f"Hasil Prediksi: 🔴 {severity}")

        st.write(
            f"**Confidence Score (Akurasi Prediksi):** {confidence*100:.2f}%"
        )

        st.markdown("---")
        st.subheader("Rekomendasi Perawatan")
        st.write(recommendation[severity])

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
