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
    "Aman":
    """
    ✅ Kulit Anda dalam kondisi bersih, sehat, dan bebas jerawat aktif!
    
    ✅ Lanjutkan perawatan harian dasar: cuci muka 2x sehari dengan pembersih lembut.
    
    ✅ Gunakan pelembap (moisturizer) ringan dan tabir surya (sunscreen) setiap hari untuk menjaga skin barrier.
    """,

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

    "Berat":
    """
    ⚠️ Tingkat keparahan jerawat Anda cukup tinggi (Berat).
    
    ⚠️ Sangat disarankan untuk berkonsultasi langsung dengan dokter spesialis kulit (dermatolog) untuk mendapatkan pengobatan medis yang tepat (seperti retinoid topikal, antibiotik, atau terapi khusus).
    """,
    
    "Parah":  # Fallback jika model lama masih terbaca
    """
    ⚠️ Tingkat keparahan jerawat Anda cukup tinggi.
    
    ⚠️ Sangat disarankan untuk berkonsultasi langsung dengan dokter spesialis kulit (dermatolog).
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

        # --- Face / Skin Detection ---
        # Convert PIL to BGR OpenCV format
        opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
        
        # Load Haar Cascades
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Detect Frontal Face
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        is_profile = False
        is_closeup = False
        
        # Jika tidak terdeteksi wajah tampak depan, coba deteksi wajah tampak samping (profile/pipi)
        if len(faces) == 0:
            faces_profile = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces_profile) > 0:
                faces = faces_profile
                is_profile = True
            else:
                # Coba balik gambar secara horizontal untuk mendeteksi profil sisi satunya (karena Cascade hanya dilatih satu sisi secara default)
                flipped_gray = cv2.flip(gray, 1)
                faces_flipped = profile_cascade.detectMultiScale(flipped_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                if len(faces_flipped) > 0:
                    h_img, w_img = gray.shape
                    faces = []
                    for (x, y, w, h) in faces_flipped:
                        # Kembalikan koordinat ke posisi asli sebelum dibalik
                        faces.append([w_img - x - w, y, w, h])
                    is_profile = True
                    
        # Jika masih tidak terdeteksi wajah, cek apakah ini foto close-up area kulit/pipi
        if len(faces) == 0:
            hsv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2HSV)
            # Batas warna kulit manusia dalam ruang warna HSV
            lower_skin1 = np.array([0, 15, 50], dtype=np.uint8)
            upper_skin1 = np.array([25, 255, 255], dtype=np.uint8)
            lower_skin2 = np.array([165, 15, 50], dtype=np.uint8)
            upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
            
            mask1 = cv2.inRange(hsv_img, lower_skin1, upper_skin1)
            mask2 = cv2.inRange(hsv_img, lower_skin2, upper_skin2)
            skin_mask = cv2.bitwise_or(mask1, mask2)
            
            # Hitung persentase piksel warna kulit
            skin_pixels = cv2.countNonZero(skin_mask)
            total_pixels = opencv_img.shape[0] * opencv_img.shape[1]
            skin_ratio = (skin_pixels / total_pixels) * 100
            
            # Jika > 45% gambar adalah area kulit, loloskan sebagai close-up pipi/wajah
            if skin_ratio > 45.0:
                is_closeup = True
                
        # Validasi akhir jika bukan wajah dan bukan pula area kulit
        if len(faces) == 0 and not is_closeup:
            st.image(
                img,
                caption="Gambar Input",
                use_container_width=True
            )
            st.error("❌ Area Wajah atau Kulit Pipi tidak terdeteksi!")
            st.warning("Mohon ambil atau unggah foto wajah (tampak depan/samping) atau foto close-up area kulit pipi Anda secara jelas dengan pencahayaan yang cukup.")
            st.stop()
            
        # Umpan Balik Visual UI
        if is_closeup:
            st.image(
                img,
                caption="Close-up Area Kulit/Pipi Terdeteksi (Valid)",
                use_container_width=True
            )
        else:
            # Gambar kotak hijau untuk wajah tampak depan atau samping
            img_for_display = opencv_img.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(img_for_display, (x, y), (x+w, y+h), (0, 255, 0), 4)
            img_with_box = Image.fromarray(cv2.cvtColor(img_for_display, cv2.COLOR_BGR2RGB))
            
            caption_text = "Wajah Samping/Pipi Terdeteksi" if is_profile else "Wajah Depan Terdeteksi"
            st.image(
                img_with_box,
                caption=f"{caption_text} (Kotak Hijau)",
                use_container_width=True
            )

        with st.spinner("Menganalisis tingkat keparahan jerawat menggunakan YOLOv8..."):
            if is_closeup:
                # Jika close-up kulit, gunakan gambar penuh langsung (karena sudah fokus ke kulit)
                cropped_face = img
            else:
                # Jika wajah depan/samping, potong area wajah tersebut dengan margin 15%
                x, y, w, h = faces[0]
                pad_w = int(w * 0.15)
                pad_h = int(h * 0.15)
                
                img_w, img_h = img.size
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(img_w, x + w + pad_w)
                y2 = min(img_h, y + h + pad_h)
                
                cropped_face = img.crop((x1, y1, x2, y2))
            
            # Jalankan prediksi
            results = model(cropped_face, verbose=False)
            
            # Mendapatkan hasil probabilitas
            probs = results[0].probs
            
            # Indeks kelas terbaik dan confidence score
            idx = probs.top1
            confidence = probs.top1conf.item()
            
            # Mengambil nama kelas asli (misal: "0_Aman", "1_Ringan", dll.)
            severity_raw = results[0].names[idx]
            # Bersihkan prefix angka (misal: "0_Aman" -> "Aman")
            severity = severity_raw.split("_", 1)[1] if "_" in severity_raw else severity_raw

        # Menampilkan Hasil
        st.success("Analisis Selesai!")
        
        # Berikan warna indikator berdasarkan kelas
        if severity == "Aman":
            st.subheader(f"Hasil Prediksi: 🔵 {severity} (Sehat)")
        elif severity == "Ringan":
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
