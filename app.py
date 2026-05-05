import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import pandas as pd
from datetime import datetime

# ── Konfigurasi halaman ──
st.set_page_config(
    page_title="SPK Deteksi Dini Kanker Kulit",
    page_icon="🔬",
    layout="wide"
)

# ── Load model & label ──
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_final.h5')
    with open('label_map.json') as f:
        label_map = json.load(f)
    return model, label_map

model, label_map = load_model()

# ── Keterangan klinis per kelas ──
info_kelas = {
    'nv': {'nama': 'Melanocytic Nevi', 'status': 'Jinak',
           'rekomendasi': 'Lesi ini bersifat jinak (tahi lalat biasa). Pantau secara berkala menggunakan kriteria ABCDE. Konsultasikan ke dokter jika ada perubahan ukuran, bentuk, atau warna.'},
    'mel': {'nama': 'Melanoma', 'status': 'Ganas',
            'rekomendasi': 'SEGERA konsultasikan ke dokter spesialis kulit (dermatologis). Melanoma adalah kanker kulit paling berbahaya. Deteksi dini sangat menentukan keberhasilan pengobatan.'},
    'bkl': {'nama': 'Benign Keratosis-like Lesions', 'status': 'Jinak',
            'rekomendasi': 'Lesi ini umumnya bersifat jinak. Disarankan pemeriksaan klinis untuk memastikan diagnosis dan memantau perkembangan lesi.'},
    'bcc': {'nama': 'Basal Cell Carcinoma', 'status': 'Ganas',
            'rekomendasi': 'Segera periksakan ke dokter spesialis kulit. Meskipun jarang bermetastasis, BCC perlu penanganan medis segera untuk mencegah kerusakan jaringan lebih lanjut.'},
    'akiec': {'nama': 'Actinic Keratoses', 'status': 'Prakanker',
              'rekomendasi': 'Actinic Keratoses adalah lesi prakanker akibat paparan sinar UV. Disarankan segera berkonsultasi dengan dokter spesialis kulit untuk penanganan dini.'},
    'vasc': {'nama': 'Vascular Lesions', 'status': 'Jinak',
             'rekomendasi': 'Lesi vaskular umumnya bersifat jinak. Konsultasikan ke dokter untuk evaluasi lebih lanjut jika lesi membesar atau berdarah.'},
    'df': {'nama': 'Dermatofibroma', 'status': 'Jinak',
           'rekomendasi': 'Dermatofibroma adalah benjolan jinak pada kulit. Umumnya tidak memerlukan pengobatan kecuali mengganggu atau ada perubahan yang mencurigakan.'}
}

# ── Inisialisasi riwayat ──
if 'riwayat' not in st.session_state:
    st.session_state.riwayat = []

# ── Sidebar ──
with st.sidebar:
    st.markdown("### SPK Deteksi Dini Kanker Kulit")
    st.caption("MobileNetV2 + Streamlit")
    st.divider()
    halaman = st.radio("Menu", ["Deteksi Lesi Kulit",
                                 "Riwayat Prediksi",
                                 "Tentang Sistem"])
    st.divider()
    st.caption("v1.0 — Tugas Akhir IF 2025")

# ══════════════════════════════
# HALAMAN 1: DETEKSI
# ══════════════════════════════
if halaman == "Deteksi Lesi Kulit":
    st.title("Deteksi Lesi Kulit")
    st.caption("Unggah citra dermoskopi untuk dianalisis oleh model MobileNetV2")
    st.divider()

    uploaded = st.file_uploader(
        "Unggah Citra Lesi Kulit",
        type=['jpg', 'jpeg', 'png'],
        help="Format: JPG, JPEG, PNG — Maks. 5 MB"
    )

    if uploaded:
        # Validasi ukuran
        if uploaded.size > 5 * 1024 * 1024:
            st.error("Ukuran berkas melebihi batas 5 MB. Silakan unggah citra lain.")
        else:
            img = Image.open(uploaded).convert('RGB')
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(img, caption="Pratinjau Citra", use_column_width=True)

            with col2:
                if st.button("Analisis Sekarang", type="primary", use_container_width=True):
                    with st.spinner("Menganalisis citra..."):
                        # Preprocessing
                        img_resized = img.resize((224, 224))
                        img_array = np.array(img_resized) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)

                        # Prediksi
                        pred = model.predict(img_array, verbose=0)[0]
                        idx = int(np.argmax(pred))
                        kode = label_map[str(idx)]
                        info = info_kelas[kode]
                        prob = float(pred[idx]) * 100

                        # Tampilkan hasil
                        st.success("Analisis selesai!")
                        st.metric("Prediksi Utama", info['nama'])
                        st.metric("Probabilitas", f"{prob:.1f}%")

                        status = info['status']
                        if status == 'Ganas':
                            st.error(f"Status: {status}")
                        elif status == 'Prakanker':
                            st.warning(f"Status: {status}")
                        else:
                            st.success(f"Status: {status}")

                        # Bar chart probabilitas
                        st.markdown("**Distribusi Probabilitas 7 Kelas**")
                        prob_df = pd.DataFrame({
                            'Kelas': [label_map[str(i)] for i in range(7)],
                            'Probabilitas (%)': [round(float(pred[i])*100, 2) for i in range(7)]
                        }).sort_values('Probabilitas (%)', ascending=False)
                        st.bar_chart(prob_df.set_index('Kelas'))

                        # Rekomendasi medis
                        st.info(f"**Rekomendasi Medis:** {info['rekomendasi']}")

                        # Simpan ke riwayat
                        st.session_state.riwayat.append({
                            'Waktu': datetime.now().strftime("%d %b %Y, %H:%M"),
                            'Prediksi': info['nama'],
                            'Probabilitas': f"{prob:.1f}%",
                            'Status': status
                        })

        if st.button("Reset", use_container_width=True):
            st.rerun()

    st.divider()
    st.caption("Hasil analisis ini bersifat sebagai alat bantu skrining awal (second opinion) dan bukan pengganti diagnosis klinis oleh dokter spesialis kulit.")

# ══════════════════════════════
# HALAMAN 2: RIWAYAT
# ══════════════════════════════
elif halaman == "Riwayat Prediksi":
    st.title("Riwayat Prediksi")
    st.caption("Seluruh sesi pemeriksaan yang telah dilakukan")
    st.divider()

    if st.session_state.riwayat:
        df_riwayat = pd.DataFrame(st.session_state.riwayat)
        st.dataframe(df_riwayat, use_container_width=True)
        if st.button("Hapus Semua Riwayat"):
            st.session_state.riwayat = []
            st.rerun()
    else:
        st.info("Belum ada riwayat prediksi. Mulai deteksi di menu Deteksi Lesi Kulit.")

# ══════════════════════════════
# HALAMAN 3: TENTANG
# ══════════════════════════════
elif halaman == "Tentang Sistem":
    st.title("Tentang Sistem")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Nama Sistem", "SPK Deteksi Dini Kanker Kulit")
        st.metric("Arsitektur Model", "MobileNetV2 + Transfer Learning")
    with col2:
        st.metric("Dataset", "HAM10000 (10.015 citra, 7 kelas)")
        st.metric("Framework", "Python · TensorFlow · Streamlit")
