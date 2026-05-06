"""
SPK Deteksi Dini Kanker Kulit
Menggunakan MobileNetV2 (HAM10000) + Streamlit
Deployed on Streamlit Community Cloud
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="SPK Deteksi Dini Kanker Kulit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESKRIPSI 7 KELAS LESI KULIT (HAM10000)
# ============================================================
CLASS_INFO = {
    "akiec": {
        "name": "Actinic Keratoses",
        "kategori": "Pra-kanker",
        "warna": "🟠",
        "deskripsi": (
            "Lesi pra-kanker akibat paparan sinar matahari berlebih. "
            "Berpotensi berkembang menjadi karsinoma sel skuamosa."
        ),
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "kategori": "Ganas (Kanker)",
        "warna": "🔴",
        "deskripsi": (
            "Kanker kulit paling umum, tumbuh lambat dan jarang menyebar. "
            "Tetap perlu penanganan medis untuk mencegah kerusakan jaringan."
        ),
    },
    "bkl": {
        "name": "Benign Keratosis",
        "kategori": "Jinak",
        "warna": "🟢",
        "deskripsi": (
            "Lesi jinak meliputi seborrheic keratoses dan solar lentigo. "
            "Umumnya tidak berbahaya namun perlu dibedakan dari melanoma."
        ),
    },
    "df": {
        "name": "Dermatofibroma",
        "kategori": "Jinak",
        "warna": "🟢",
        "deskripsi": (
            "Benjolan kulit jinak akibat reaksi terhadap luka kecil "
            "atau gigitan serangga. Tidak berbahaya."
        ),
    },
    "mel": {
        "name": "Melanoma",
        "kategori": "Ganas (Kanker)",
        "warna": "🔴",
        "deskripsi": (
            "Kanker kulit paling agresif, dapat menyebar cepat ke organ lain. "
            "Deteksi dini sangat krusial untuk prognosis yang baik."
        ),
    },
    "nv": {
        "name": "Melanocytic Nevi",
        "kategori": "Jinak",
        "warna": "🟢",
        "deskripsi": (
            "Tahi lalat biasa. Mayoritas tidak berbahaya, namun perlu "
            "dipantau jika berubah ukuran, warna, atau bentuk."
        ),
    },
    "vasc": {
        "name": "Vascular Lesions",
        "kategori": "Jinak",
        "warna": "🟢",
        "deskripsi": (
            "Lesi pembuluh darah seperti hemangioma dan angioma ceri. "
            "Umumnya jinak dan tidak memerlukan perawatan."
        ),
    },
}

# ============================================================
# LOAD MODEL & LABEL (cache supaya hanya dimuat sekali)
# ============================================================
@st.cache_resource(show_spinner="🔄 Memuat model MobileNetV2...")
def load_model_and_labels():
    """Load model H5 dan label map. Dijalankan sekali per session."""
    model = tf.keras.models.load_model("model_final.h5")
    with open("label_map.json", "r") as f:
        label_map = json.load(f)
    # Pastikan format konsisten: {index: kode_kelas}
    # Contoh: {"0": "akiec", "1": "bcc", ...}
    return model, label_map


# ============================================================
# PREPROCESSING GAMBAR
# ============================================================
def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Konversi PIL Image -> numpy array siap input MobileNetV2.
    - Resize ke 224x224
    - Konversi ke RGB (jaga-jaga input PNG dengan alpha channel)
    - Normalisasi 0-1
    - Expand dim ke (1, 224, 224, 3)
    """
    image = image.convert("RGB").resize(target_size)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ============================================================
# SIDEBAR — INFO APLIKASI
# ============================================================
with st.sidebar:
    st.markdown("## 🔬 Tentang Aplikasi")
    st.markdown(
        """
        Sistem Pendukung Keputusan untuk **deteksi dini kanker kulit**
        menggunakan model deep learning **MobileNetV2** yang dilatih
        pada dataset **HAM10000**.
        
        **7 kelas lesi kulit:**
        - 🔴 Melanoma (mel)
        - 🔴 Basal Cell Carcinoma (bcc)
        - 🟠 Actinic Keratoses (akiec)
        - 🟢 Melanocytic Nevi (nv)
        - 🟢 Benign Keratosis (bkl)
        - 🟢 Dermatofibroma (df)
        - 🟢 Vascular Lesions (vasc)
        """
    )
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.warning(
        "Aplikasi ini adalah **alat bantu skripsi** dan **BUKAN pengganti "
        "diagnosis medis profesional**. Hasil prediksi harus selalu "
        "dikonfirmasi oleh dokter spesialis kulit (dermatolog)."
    )
    st.markdown("---")
    st.caption("📚 Skripsi — MobileNetV2 + Streamlit")


# ============================================================
# MAIN PAGE
# ============================================================
st.title("🔬 SPK Deteksi Dini Kanker Kulit")
st.markdown(
    "**Unggah gambar lesi kulit** untuk mendapatkan prediksi klasifikasi "
    "menggunakan model MobileNetV2 yang telah dilatih pada dataset HAM10000."
)

# Load model di awal agar error muncul lebih cepat kalau ada masalah
try:
    model, label_map = load_model_and_labels()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# Layout 2 kolom
col_upload, col_result = st.columns([1, 1.3])

with col_upload:
    st.subheader("📤 Unggah Gambar")
    uploaded_file = st.file_uploader(
        "Pilih file gambar (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        help="Gunakan gambar lesi kulit yang jelas dan tidak buram.",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(
            image,
            caption=f"Gambar yang diunggah: {uploaded_file.name}",
            use_container_width=True,
        )

with col_result:
    st.subheader("📊 Hasil Prediksi")

    if uploaded_file is None:
        st.info("👈 Silakan unggah gambar terlebih dahulu di kolom kiri.")
    else:
        with st.spinner("🧠 Menganalisis gambar..."):
            # Preprocess + Predict
            img_array = preprocess_image(image)
            predictions = model.predict(img_array, verbose=0)[0]

        # Mapping index -> nama kelas
        # label_map bisa berupa {"0": "akiec"} atau {"akiec": 0}
        # Kita bikin keduanya kompatibel:
        if all(isinstance(v, int) for v in label_map.values()):
            # Format {"akiec": 0, "bcc": 1, ...} -> balik
            idx_to_label = {v: k for k, v in label_map.items()}
        else:
            # Format {"0": "akiec", "1": "bcc", ...}
            idx_to_label = {int(k): v for k, v in label_map.items()}

        # Top-3 prediksi
        top3_idx = predictions.argsort()[-3:][::-1]

        # ---------- Hasil Utama ----------
        top1_idx = top3_idx[0]
        top1_label = idx_to_label[top1_idx]
        top1_confidence = predictions[top1_idx]
        top1_info = CLASS_INFO.get(top1_label, {})

        st.markdown(
            f"### {top1_info.get('warna', '⚪')} "
            f"{top1_info.get('name', top1_label.upper())}"
        )
        st.markdown(f"**Kategori:** {top1_info.get('kategori', '-')}")
        st.progress(float(top1_confidence))
        st.markdown(f"**Tingkat Keyakinan:** `{top1_confidence*100:.2f}%`")

        with st.expander("ℹ️ Penjelasan kelas ini"):
            st.write(top1_info.get("deskripsi", "Tidak ada deskripsi tersedia."))

        # ---------- Top-3 Confidence Chart ----------
        st.markdown("---")
        st.markdown("#### 📈 Top-3 Prediksi")
        df_top3 = pd.DataFrame(
            {
                "Kelas": [
                    CLASS_INFO.get(idx_to_label[i], {}).get(
                        "name", idx_to_label[i]
                    )
                    for i in top3_idx
                ],
                "Confidence (%)": [predictions[i] * 100 for i in top3_idx],
            }
        )
        st.dataframe(
            df_top3.style.format({"Confidence (%)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(df_top3.set_index("Kelas"))

        # ---------- Probabilitas Semua Kelas ----------
        with st.expander("🔍 Lihat probabilitas semua 7 kelas"):
            df_all = pd.DataFrame(
                {
                    "Kode": [idx_to_label[i] for i in range(len(predictions))],
                    "Nama Kelas": [
                        CLASS_INFO.get(idx_to_label[i], {}).get(
                            "name", idx_to_label[i]
                        )
                        for i in range(len(predictions))
                    ],
                    "Probabilitas (%)": [p * 100 for p in predictions],
                }
            ).sort_values("Probabilitas (%)", ascending=False)
            st.dataframe(
                df_all.style.format({"Probabilitas (%)": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

        # ---------- Disclaimer ----------
        st.markdown("---")
        st.error(
            "⚠️ **Penting:** Hasil ini hanya untuk tujuan **edukatif & skripsi**. "
            "Konsultasikan dengan **dokter spesialis kulit** untuk diagnosis akurat."
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "🎓 Skripsi — Sistem Pendukung Keputusan Deteksi Dini Kanker Kulit | "
    "Model: MobileNetV2 | Dataset: HAM10000 | Framework: TensorFlow + Streamlit"
)
