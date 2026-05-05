# 🔬 SPK Deteksi Dini Kanker Kulit

Sistem Pendukung Keputusan untuk deteksi dini kanker kulit menggunakan **MobileNetV2** yang dilatih pada dataset **HAM10000**.

## 🌐 Live Demo

Aplikasi ini di-deploy di **Streamlit Community Cloud**:
👉 [Buka Aplikasi](https://share.streamlit.io)

## 📸 Fitur

- Upload gambar lesi kulit (JPG, JPEG, PNG)
- Klasifikasi 7 kelas lesi kulit:
  - 🔴 **Melanoma (mel)** — Ganas
  - 🔴 **Basal Cell Carcinoma (bcc)** — Ganas
  - 🟠 **Actinic Keratoses (akiec)** — Pra-kanker
  - 🟢 **Melanocytic Nevi (nv)** — Jinak
  - 🟢 **Benign Keratosis (bkl)** — Jinak
  - 🟢 **Dermatofibroma (df)** — Jinak
  - 🟢 **Vascular Lesions (vasc)** — Jinak
- Top-3 prediksi dengan confidence score
- Visualisasi probabilitas semua kelas
- Penjelasan singkat tiap kategori lesi

## 🛠️ Tech Stack

| Komponen | Versi |
|----------|-------|
| Python | 3.11 |
| TensorFlow (CPU) | 2.15.0 |
| Streamlit | 1.32.0 |
| NumPy | 1.26.4 |
| Pillow | 10.4.0 |
| Pandas | 2.2.2 |

## 📂 Struktur Project

```
spk-kanker-kulit/
├── app.py              # Main Streamlit app
├── model_final.h5      # Model MobileNetV2 (25 MB)
├── label_map.json      # Mapping index → nama kelas
├── requirements.txt    # Dependencies
├── .gitignore
└── README.md
```

## 🚀 Cara Menjalankan Lokal

```bash
# 1. Clone repository
git clone https://github.com/USERNAME/spk-kanker-kulit.git
cd spk-kanker-kulit

# 2. Buat virtual environment
python3 -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan
streamlit run app.py
```

## 📊 Dataset

Model dilatih pada **HAM10000** ("Human Against Machine with 10000 training images") — dataset publik 10,015 gambar dermatoscopic dari berbagai populasi.

Sumber: [Harvard Dataverse — HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

## ⚠️ Disclaimer

> Aplikasi ini adalah **alat bantu edukasi & penelitian skripsi**, **BUKAN pengganti diagnosis medis profesional**. Hasil prediksi harus selalu dikonfirmasi oleh **dokter spesialis kulit (dermatolog)**.

## 🎓 Konteks

Skripsi: *Sistem Pendukung Keputusan Deteksi Dini Kanker Kulit menggunakan MobileNetV2*

## 📜 Lisensi

Untuk keperluan akademis. Dataset HAM10000 mengikuti lisensi aslinya.
