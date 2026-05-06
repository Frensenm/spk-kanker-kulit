"""
SPK Deteksi Dini Kanker Kulit
MobileNetV2 + HAM10000 + Streamlit
"""

import json
import io
import base64
from datetime import datetime

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
# CUSTOM CSS — Dark Theme sesuai referensi
# ============================================================
st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:    #0f1117;
    --bg-secondary:  #1a1d27;
    --bg-card:       #21242f;
    --bg-hover:      #2a2d3a;
    --accent-blue:   #3b82f6;
    --accent-red:    #ef4444;
    --accent-orange: #f97316;
    --accent-green:  #22c55e;
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #64748b;
    --border:        #2d3148;
    --radius:        10px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1100px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding: 1.5rem 1.2rem !important; }

/* ── App branding in sidebar ── */
.app-brand { margin-bottom: 1.5rem; }
.app-brand h2 {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.3;
    margin: 0 0 2px 0;
}
.app-brand p {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin: 0;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Sidebar menu label ── */
.menu-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin: 0 0 0.5rem 0;
}

/* ── Sidebar nav buttons ── */
div[data-testid="stSidebar"] .stButton button {
    width: 100% !important;
    text-align: left !important;
    background: transparent !important;
    border: none !important;
    border-radius: var(--radius) !important;
    color: var(--text-secondary) !important;
    font-size: 0.875rem !important;
    font-weight: 400 !important;
    padding: 0.55rem 0.9rem !important;
    margin-bottom: 2px !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stSidebar"] .stButton button:hover {
    background-color: var(--bg-hover) !important;
    color: var(--text-primary) !important;
}

/* ── Active nav button (via key hack) ── */
.nav-active button {
    background-color: var(--accent-blue) !important;
    color: #fff !important;
    font-weight: 600 !important;
}

/* ── Disclaimer box in sidebar ── */
.disclaimer-box {
    background-color: #2a1f0e;
    border: 1px solid #78350f;
    border-radius: var(--radius);
    padding: 0.9rem;
    margin-top: 0.5rem;
}
.disclaimer-box p {
    font-size: 0.8rem;
    color: #fbbf24;
    margin: 0;
    line-height: 1.6;
}
.disclaimer-box strong { color: #f59e0b; }

/* ── Page titles ── */
.page-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 2px 0;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 0.875rem;
    color: var(--text-muted);
    margin: 0 0 1.8rem 0;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
}

/* ── Upload area ── */
div[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 2rem !important;
}
div[data-testid="stFileUploader"] label {
    color: var(--text-secondary) !important;
    font-size: 0.875rem !important;
}

/* ── Main action button ── */
.stButton button[kind="primary"], button[data-testid*="primary"] {
    background-color: var(--accent-blue) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton button[kind="primary"]:hover {
    background-color: #2563eb !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(59,130,246,0.4) !important;
}

/* ── Secondary button ── */
.stButton button[kind="secondary"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.2rem !important;
    width: 100% !important;
}

/* ── Disclaimer text ── */
.upload-disclaimer {
    font-size: 0.78rem;
    color: var(--text-muted);
    padding: 0.75rem 0.9rem;
    background: var(--bg-card);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    line-height: 1.6;
    margin-top: 0.5rem;
}

/* ── Result card ── */
.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
    margin-bottom: 0.75rem;
}
.result-card-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}
.result-class-name {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--accent-red);
    margin: 0 0 4px 0;
}
.result-class-name.jinak { color: var(--accent-green); }
.result-class-name.prakanker { color: var(--accent-orange); }

.badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 999px;
}
.badge-ganas  { background: #450a0a; color: #fca5a5; }
.badge-prakanker { background: #431407; color: #fdba74; }
.badge-jinak  { background: #052e16; color: #86efac; }

.confidence-value {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text-primary);
}

/* ── Probability bar list ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    font-size: 0.82rem;
}
.prob-label { width: 200px; color: var(--text-secondary); flex-shrink: 0; }
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: var(--bg-hover);
    border-radius: 999px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: var(--accent-blue);
    transition: width 0.6s ease;
}
.prob-bar-fill.top { background: var(--accent-red); }
.prob-bar-fill.orange { background: var(--accent-orange); }
.prob-bar-fill.green { background: var(--accent-green); }
.prob-pct {
    width: 46px;
    text-align: right;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-secondary);
    flex-shrink: 0;
}

/* ── Rekomendasi medis ── */
.rekomendasi-box {
    background: #0f1e3d;
    border: 1px solid #1e3a8a;
    border-radius: var(--radius);
    padding: 1rem;
    margin-top: 0.5rem;
}
.rekomendasi-box p { font-size: 0.83rem; color: #93c5fd; margin: 0; line-height: 1.7; }

/* ── Action buttons row ── */
.btn-row { display: flex; gap: 10px; margin-top: 1rem; }

/* ── History table ── */
.stDataFrame { border-radius: var(--radius) !important; }
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
}

/* ── Progress bar override ── */
.stProgress > div > div { background-color: var(--accent-blue) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Streamlit image caption ── */
div[data-testid="caption"] { color: var(--text-muted) !important; font-size: 0.78rem !important; }

/* ── Remove default radio button styling for nav ── */
div[data-testid="stRadio"] > label { display: none; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DESKRIPSI 7 KELAS LESI KULIT
# ============================================================
CLASS_INFO = {
    "akiec": {
        "name": "Actinic Keratoses",
        "short": "akiec",
        "kategori": "Pra-kanker",
        "badge_class": "badge-prakanker",
        "name_class": "prakanker",
        "warna": "🟠",
        "rekomendasi": "Lesi pra-kanker akibat paparan sinar UV berlebih. Segera periksakan ke dokter spesialis kulit untuk penanganan dini sebelum berkembang menjadi karsinoma sel skuamosa.",
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "short": "bcc",
        "kategori": "Ganas (Kanker)",
        "badge_class": "badge-ganas",
        "name_class": "ganas",
        "warna": "🔴",
        "rekomendasi": "Kanker kulit yang perlu penanganan medis segera. Meskipun jarang bermetastasis, BCC dapat merusak jaringan sekitar jika dibiarkan. Segera konsultasikan ke dermatologis.",
    },
    "bkl": {
        "name": "Benign Keratosis",
        "short": "bkl",
        "kategori": "Jinak",
        "badge_class": "badge-jinak",
        "name_class": "jinak",
        "warna": "🟢",
        "rekomendasi": "Lesi jinak meliputi seborrheic keratoses dan solar lentigo. Umumnya tidak berbahaya. Tetap disarankan pemeriksaan klinis untuk konfirmasi diagnosis.",
    },
    "df": {
        "name": "Dermatofibroma",
        "short": "df",
        "kategori": "Jinak",
        "badge_class": "badge-jinak",
        "name_class": "jinak",
        "warna": "🟢",
        "rekomendasi": "Benjolan jinak pada kulit. Umumnya tidak memerlukan pengobatan kecuali mengganggu aktivitas atau terdapat perubahan yang mencurigakan.",
    },
    "mel": {
        "name": "Melanoma",
        "short": "mel",
        "kategori": "Ganas (Kanker)",
        "badge_class": "badge-ganas",
        "name_class": "ganas",
        "warna": "🔴",
        "rekomendasi": "Melanoma adalah kanker kulit paling berbahaya yang berasal dari sel melanosit. Deteksi dini sangat menentukan keberhasilan pengobatan. Hasil ini bukan diagnosis final — segera periksakan ke dermatologis untuk tindak lanjut klinis.",
    },
    "nv": {
        "name": "Melanocytic Nevi",
        "short": "nv",
        "kategori": "Jinak",
        "badge_class": "badge-jinak",
        "name_class": "jinak",
        "warna": "🟢",
        "rekomendasi": "Tahi lalat biasa yang bersifat jinak. Pantau secara berkala menggunakan kriteria ABCDE (Asymmetry, Border, Color, Diameter, Evolution). Konsultasikan jika ada perubahan.",
    },
    "vasc": {
        "name": "Vascular Lesions",
        "short": "vasc",
        "kategori": "Jinak",
        "badge_class": "badge-jinak",
        "name_class": "jinak",
        "warna": "🟢",
        "rekomendasi": "Lesi pembuluh darah seperti hemangioma dan angioma. Umumnya jinak dan tidak memerlukan perawatan, namun konsultasikan ke dokter jika membesar atau berdarah.",
    },
}

# ============================================================
# SESSION STATE INIT
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "Deteksi Lesi Kulit"
if "riwayat" not in st.session_state:
    st.session_state.riwayat = []
if "hasil" not in st.session_state:
    st.session_state.hasil = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    model = tf.keras.models.load_model("model_final.h5")
    with open("label_map.json", "r") as f:
        label_map = json.load(f)
    return model, label_map

def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    image = image.convert("RGB").resize(target_size)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def image_to_base64(img: Image.Image, size=(60, 60)) -> str:
    img_thumb = img.copy().convert("RGB")
    img_thumb.thumbnail(size)
    buf = io.BytesIO()
    img_thumb.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()

try:
    model, label_map = load_model_and_labels()
    if all(isinstance(v, int) for v in label_map.values()):
        idx_to_label = {v: k for k, v in label_map.items()}
    else:
        idx_to_label = {int(k): v for k, v in label_map.items()}
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="app-brand">
        <h2>SPK Deteksi Dini Kanker Kulit</h2>
        <p>MobileNetV2 + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="menu-label">MENU</p>', unsafe_allow_html=True)

    pages = ["Deteksi Lesi Kulit", "Hasil Prediksi", "Riwayat Prediksi", "Tentang Sistem"]
    icons  = ["🔍", "📊", "📋", "ℹ️"]

    for pg, ic in zip(pages, icons):
        is_active = st.session_state.page == pg
        container = st.container()
        if is_active:
            container.markdown('<div class="nav-active">', unsafe_allow_html=True)
        if container.button(f"{ic}  {pg}", key=f"nav_{pg}", use_container_width=True):
            st.session_state.page = pg
            st.rerun()
        if is_active:
            container.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="menu-label">⚠️ Disclaimer</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer-box">
        <p>Aplikasi ini adalah <strong>alat bantu skripsi</strong> dan <strong>BUKAN pengganti
        diagnosis medis profesional</strong>. Hasil prediksi harus selalu
        dikonfirmasi oleh dokter spesialis kulit (dermatolog).</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem;color:#475569;text-align:center">📚 Skripsi — MobileNetV2 + Streamlit</p>', unsafe_allow_html=True)

# ============================================================
# HALAMAN 1 — DETEKSI LESI KULIT
# ============================================================
if st.session_state.page == "Deteksi Lesi Kulit":

    st.markdown('<h1 class="page-title">Deteksi Lesi Kulit</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Unggah citra dermoskopi untuk dianalisis oleh model MobileNetV2</p>', unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"❌ Gagal memuat model: {model_error}")
        st.stop()

    st.markdown('<p class="section-label">UNGGAH CITRA</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag and drop citra di sini",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Format: JPG, JPEG, PNG — Maks. 5 MB",
    )

    if uploaded_file is not None:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("❌ Ukuran file melebihi 5 MB. Silakan unggah file yang lebih kecil.")
            uploaded_file = None
        else:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            st.image(image, caption=f"📎 {uploaded_file.name}", use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        analisis_btn = st.button("Analisis Sekarang", type="primary", use_container_width=True, key="btn_analisis")

    with col_b:
        reset_btn = st.button("Reset", type="secondary", use_container_width=True, key="btn_reset")

    if reset_btn:
        st.session_state.uploaded_image = None
        st.session_state.hasil = None
        st.rerun()

    if analisis_btn:
        if st.session_state.uploaded_image is None:
            st.warning("⚠️ Silakan unggah gambar terlebih dahulu.")
        else:
            with st.spinner("🧠 Menganalisis citra..."):
                img_array = preprocess_image(st.session_state.uploaded_image)
                predictions = model.predict(img_array, verbose=0)[0]

            top_idx = int(np.argmax(predictions))
            top_label = idx_to_label[top_idx]
            top_confidence = float(predictions[top_idx])
            top_info = CLASS_INFO.get(top_label, {})

            # Susun semua probabilitas
            all_probs = [
                {
                    "label": idx_to_label[i],
                    "name": CLASS_INFO.get(idx_to_label[i], {}).get("name", idx_to_label[i]),
                    "prob": float(predictions[i]),
                    "kategori": CLASS_INFO.get(idx_to_label[i], {}).get("kategori", ""),
                }
                for i in range(len(predictions))
            ]
            all_probs.sort(key=lambda x: x["prob"], reverse=True)

            st.session_state.hasil = {
                "label": top_label,
                "name": top_info.get("name", top_label),
                "kategori": top_info.get("kategori", "-"),
                "badge_class": top_info.get("badge_class", "badge-jinak"),
                "name_class": top_info.get("name_class", "jinak"),
                "confidence": top_confidence,
                "rekomendasi": top_info.get("rekomendasi", ""),
                "all_probs": all_probs,
                "image": st.session_state.uploaded_image,
                "waktu": datetime.now().strftime("%d %b %Y, %H:%M"),
            }

            # Simpan ke riwayat
            st.session_state.riwayat.append({
                "waktu": datetime.now().strftime("%d %b %Y, %H:%M"),
                "image_b64": image_to_base64(st.session_state.uploaded_image),
                "prediksi": f"{top_info.get('name', top_label)} ({top_label})",
                "prob": f"{top_confidence*100:.1f}%",
                "status": top_info.get("kategori", "-"),
                "badge_class": top_info.get("badge_class", "badge-jinak"),
            })

            st.session_state.page = "Hasil Prediksi"
            st.rerun()

    st.markdown("""
    <div class="upload-disclaimer">
        Hasil analisis sistem ini bersifat sebagai alat bantu skrining awal (second opinion) dan
        bukan merupakan pengganti diagnosis klinis oleh dokter spesialis kulit.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HALAMAN 2 — HASIL PREDIKSI
# ============================================================
elif st.session_state.page == "Hasil Prediksi":

    st.markdown('<h1 class="page-title">Hasil Prediksi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Analisis selesai — berikut hasil klasifikasi model</p>', unsafe_allow_html=True)

    if st.session_state.hasil is None:
        st.info("💡 Belum ada hasil prediksi. Silakan unggah dan analisis citra terlebih dahulu.")
        if st.button("← Kembali ke Deteksi", type="primary"):
            st.session_state.page = "Deteksi Lesi Kulit"
            st.rerun()
    else:
        h = st.session_state.hasil

        col_left, col_right = st.columns([1, 1.2], gap="large")

        # ── Kolom kiri ──
        with col_left:
            st.markdown('<p class="section-label">CITRA YANG DIANALISIS</p>', unsafe_allow_html=True)
            if h["image"] is not None:
                st.image(h["image"], use_container_width=True)

            # Prediksi utama
            name_class = h.get("name_class", "jinak")
            badge_class = h.get("badge_class", "badge-jinak")

            st.markdown(f"""
            <div class="result-card" style="margin-top:1rem">
                <div class="result-card-label">Prediksi Utama</div>
                <div class="result-class-name {name_class}">{h['name']} ({h['label']})</div>
                <span class="badge {badge_class}">{h['kategori']}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-card">
                <div class="result-card-label">Probabilitas Prediksi</div>
                <div class="confidence-value">{h['confidence']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Rekomendasi medis
            st.markdown('<p class="section-label" style="margin-top:1rem">REKOMENDASI MEDIS</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="rekomendasi-box">
                <p>{h['rekomendasi']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Tombol aksi
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Periksa Citra Lain", type="secondary", use_container_width=True, key="btn_periksa_lain"):
                    st.session_state.hasil = None
                    st.session_state.uploaded_image = None
                    st.session_state.page = "Deteksi Lesi Kulit"
                    st.rerun()
            with col_btn2:
                if st.button("Lihat Riwayat", type="secondary", use_container_width=True, key="btn_lihat_riwayat"):
                    st.session_state.page = "Riwayat Prediksi"
                    st.rerun()

        # ── Kolom kanan ──
        with col_right:
            st.markdown('<p class="section-label">DISTRIBUSI PROBABILITAS 7 KELAS</p>', unsafe_allow_html=True)

            max_prob = h["all_probs"][0]["prob"]

            for i, item in enumerate(h["all_probs"]):
                pct = item["prob"] * 100
                width_pct = (item["prob"] / max_prob) * 100 if max_prob > 0 else 0

                # Warna bar berdasarkan kategori
                if i == 0:
                    bar_class = "top"
                elif "Ganas" in item["kategori"]:
                    bar_class = "top"
                elif "Pra" in item["kategori"]:
                    bar_class = "orange"
                else:
                    bar_class = "green"

                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-label">{item['name']} ({item['label']})</div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill {bar_class}" style="width:{width_pct:.1f}%"></div>
                    </div>
                    <div class="prob-pct">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# HALAMAN 3 — RIWAYAT PREDIKSI
# ============================================================
elif st.session_state.page == "Riwayat Prediksi":

    st.markdown('<h1 class="page-title">Riwayat Prediksi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Seluruh sesi pemeriksaan yang telah dilakukan</p>', unsafe_allow_html=True)

    if not st.session_state.riwayat:
        st.info("💡 Belum ada riwayat prediksi. Mulai deteksi di menu Deteksi Lesi Kulit.")
        if st.button("← Mulai Deteksi", type="primary", key="btn_mulai_deteksi"):
            st.session_state.page = "Deteksi Lesi Kulit"
            st.rerun()
    else:
        # ── Header row ──
        st.markdown("""
        <div style="display:grid;grid-template-columns:40px 150px 60px 1fr 110px 100px;
                    gap:0;border:1px solid #2d3148;border-radius:10px 10px 0 0;
                    background:#1a1d27;padding:10px 14px;margin-bottom:0">
            <span style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;color:#64748b;text-transform:uppercase">No</span>
            <span style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;color:#64748b;text-transform:uppercase">Waktu</span>
            <span style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;color:#64748b;text-transform:uppercase">Citra</span>
            <span style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;color:#64748b;text-transform:uppercase">Prediksi</span>
            <span style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;color:#64748b;text-transform:uppercase">Probabilitas</span>
            <span style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;color:#64748b;text-transform:uppercase">Status</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Data rows menggunakan Streamlit columns ──
        riwayat_reversed = list(reversed(st.session_state.riwayat))
        for i, r in enumerate(riwayat_reversed, 1):
            # Border styling: last row rounded bottom
            is_last = (i == len(riwayat_reversed))
            border_radius = "0 0 10px 10px" if is_last else "0"
            bg = "#21242f" if i % 2 == 0 else "#1e2130"

            c_no, c_waktu, c_img, c_pred, c_prob, c_status = st.columns(
                [0.4, 1.5, 0.6, 2.2, 1.1, 1.0]
            )

            # Decode base64 thumbnail → PIL Image
            try:
                img_bytes = base64.b64decode(r["image_b64"])
                thumb = Image.open(io.BytesIO(img_bytes))
            except Exception:
                thumb = None

            row_style = (
                f"background:{bg};padding:8px 4px;"
                f"border-left:1px solid #2d3148;"
                f"border-right:1px solid #2d3148;"
                f"border-bottom:1px solid #2d3148;"
                f"border-radius:{border_radius};"
            )

            with c_no:
                st.markdown(
                    f'<div style="{row_style}padding-left:14px;'
                    f'color:#64748b;font-size:0.83rem;line-height:44px">{i}</div>',
                    unsafe_allow_html=True,
                )
            with c_waktu:
                st.markdown(
                    f'<div style="{row_style}color:#94a3b8;font-size:0.8rem;'
                    f'line-height:1.4;padding-top:10px">{r["waktu"]}</div>',
                    unsafe_allow_html=True,
                )
            with c_img:
                st.markdown(f'<div style="{row_style}">', unsafe_allow_html=True)
                if thumb:
                    st.image(thumb, width=44)
                st.markdown("</div>", unsafe_allow_html=True)
            with c_pred:
                st.markdown(
                    f'<div style="{row_style}color:#f1f5f9;font-size:0.84rem;'
                    f'line-height:1.4;padding-top:10px">{r["prediksi"]}</div>',
                    unsafe_allow_html=True,
                )
            with c_prob:
                st.markdown(
                    f'<div style="{row_style}font-family:\'IBM Plex Mono\',monospace;'
                    f'color:#f1f5f9;font-size:0.84rem;line-height:44px;text-align:center">'
                    f'{r["prob"]}</div>',
                    unsafe_allow_html=True,
                )
            with c_status:
                st.markdown(
                    f'<div style="{row_style}padding-top:11px;text-align:center">'
                    f'<span class="badge {r["badge_class"]}">{r["status"]}</span></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        col_act1, col_act2, col_act3 = st.columns([1, 1, 2])
        with col_act1:
            if st.button("Hapus Riwayat", type="secondary", use_container_width=True, key="btn_hapus"):
                st.session_state.riwayat = []
                st.rerun()
        with col_act2:
            if st.button("Deteksi Baru", type="primary", use_container_width=True, key="btn_deteksi_baru"):
                st.session_state.hasil = None
                st.session_state.uploaded_image = None
                st.session_state.page = "Deteksi Lesi Kulit"
                st.rerun()

# ============================================================
# HALAMAN 4 — TENTANG SISTEM
# ============================================================
elif st.session_state.page == "Tentang Sistem":

    st.markdown('<h1 class="page-title">Tentang Sistem</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Informasi teknis dan deskripsi kelas lesi kulit</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="result-card">
            <div class="result-card-label">Nama Sistem</div>
            <div style="font-size:1rem;font-weight:600;color:#f1f5f9">SPK Deteksi Dini Kanker Kulit</div>
        </div>
        <div class="result-card">
            <div class="result-card-label">Arsitektur Model</div>
            <div style="font-size:1rem;font-weight:600;color:#f1f5f9">MobileNetV2 + Transfer Learning</div>
        </div>
        <div class="result-card">
            <div class="result-card-label">Dataset</div>
            <div style="font-size:1rem;font-weight:600;color:#f1f5f9">HAM10000 (10.015 citra, 7 kelas)</div>
        </div>
        <div class="result-card">
            <div class="result-card-label">Framework</div>
            <div style="font-size:1rem;font-weight:600;color:#f1f5f9">Python · TensorFlow · Streamlit</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-label">7 KELAS LESI KULIT</p>', unsafe_allow_html=True)
        for code, info in CLASS_INFO.items():
            st.markdown(f"""
            <div class="result-card" style="margin-bottom:6px">
                <div style="display:flex;align-items:center;justify-content:space-between">
                    <div>
                        <span style="font-size:0.85rem;font-weight:600;color:#f1f5f9">{info['warna']} {info['name']}</span>
                        <span style="font-size:0.75rem;color:#64748b;font-family:'IBM Plex Mono',monospace;margin-left:8px">({code})</span>
                    </div>
                    <span class="badge {info['badge_class']}">{info['kategori']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
