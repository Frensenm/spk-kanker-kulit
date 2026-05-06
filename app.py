"""
SPK Deteksi Dini Kanker Kulit
MobileNetV2 + HAM10000 + Streamlit
Final version — 3 halaman: Deteksi, Riwayat, Tentang
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

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SPK Deteksi Dini Kanker Kulit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════
# CSS
# ═════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg:         #0a0c12;
  --bg2:        #111420;
  --bg3:        #181c2a;
  --bg4:        #1f243a;
  --border:     #252a40;
  --border2:    #2e3550;
  --accent:     #4f7cff;
  --accent-dim: #1e2d5a;
  --red:        #ff4d6d;
  --red-dim:    #3d0f1a;
  --orange:     #ff8c42;
  --orange-dim: #3d2010;
  --green:      #3ecf8e;
  --green-dim:  #0d3326;
  --t1:         #eef0f8;
  --t2:         #8892b0;
  --t3:         #4a5270;
  --radius:     12px;
  --radius-sm:  8px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--t1) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 2rem 2.5rem 4rem !important;
  max-width: 1080px;
}

/* ── Sidebar ──────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div {
  padding: 2rem 1.4rem !important;
}

.brand-block { margin-bottom: 2.5rem; }
.brand-icon {
  font-size: 2rem;
  margin-bottom: 0.5rem;
  display: block;
}
.brand-title {
  font-family: 'Syne', sans-serif;
  font-size: 1rem;
  font-weight: 700;
  color: var(--t1);
  line-height: 1.3;
  margin: 0 0 3px;
}
.brand-sub {
  font-family: 'DM Mono', monospace;
  font-size: 0.68rem;
  color: var(--t3);
  margin: 0;
}

.nav-label {
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--t3);
  margin: 0 0 0.6rem;
}

div[data-testid="stSidebar"] .stButton button {
  width: 100% !important;
  text-align: left !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: var(--radius-sm) !important;
  color: var(--t2) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.875rem !important;
  font-weight: 400 !important;
  padding: 0.6rem 0.9rem !important;
  margin-bottom: 3px !important;
  transition: all 0.15s ease !important;
  letter-spacing: 0 !important;
}
div[data-testid="stSidebar"] .stButton button:hover {
  background: var(--bg3) !important;
  border-color: var(--border) !important;
  color: var(--t1) !important;
}
.nav-active div[data-testid="stSidebar"] .stButton button,
.nav-active .stButton button {
  background: var(--accent-dim) !important;
  border-color: var(--accent) !important;
  color: #a8c0ff !important;
  font-weight: 500 !important;
}

.disc-box {
  background: #1a1208;
  border: 1px solid #3d2e0a;
  border-radius: var(--radius-sm);
  padding: 0.85rem;
  margin-top: 0.4rem;
}
.disc-box p {
  font-size: 0.78rem;
  color: #c8a83a;
  margin: 0;
  line-height: 1.65;
}
.disc-box strong { color: #f0c040; }

/* ── Page heading ─────────────────────────────────── */
.page-heading {
  font-family: 'Syne', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  color: var(--t1);
  letter-spacing: -0.03em;
  margin: 0 0 4px;
  line-height: 1.15;
}
.page-sub {
  font-size: 0.88rem;
  color: var(--t3);
  margin: 0 0 2rem;
}

/* ── Upload zone ──────────────────────────────────── */
div[data-testid="stFileUploader"] {
  background: var(--bg3) !important;
  border: 1.5px dashed var(--border2) !important;
  border-radius: var(--radius) !important;
  transition: border-color 0.2s !important;
}
div[data-testid="stFileUploader"]:hover {
  border-color: var(--accent) !important;
}
div[data-testid="stFileUploader"] label { display: none !important; }

.upload-hint {
  text-align: center;
  padding: 0.4rem 0 0.8rem;
  font-size: 0.78rem;
  color: var(--t3);
}
.upload-hint span {
  display: inline-block;
  background: var(--bg4);
  border: 1px solid var(--border2);
  border-radius: 99px;
  padding: 2px 10px;
  font-family: 'DM Mono', monospace;
  font-size: 0.7rem;
  margin: 0 2px;
}

/* ── Buttons ──────────────────────────────────────── */
.stButton button {
  border-radius: var(--radius-sm) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.875rem !important;
  transition: all 0.18s ease !important;
  letter-spacing: 0.01em !important;
}
.stButton button[kind="primary"] {
  background: var(--accent) !important;
  border: none !important;
  color: #fff !important;
  padding: 0.65rem 1.4rem !important;
}
.stButton button[kind="primary"]:hover {
  background: #3a68f5 !important;
  box-shadow: 0 4px 20px rgba(79,124,255,0.4) !important;
  transform: translateY(-1px) !important;
}
.stButton button[kind="secondary"] {
  background: var(--bg3) !important;
  border: 1px solid var(--border2) !important;
  color: var(--t2) !important;
  padding: 0.65rem 1.4rem !important;
}
.stButton button[kind="secondary"]:hover {
  background: var(--bg4) !important;
  border-color: var(--t3) !important;
  color: var(--t1) !important;
}

/* ── Section label ────────────────────────────────── */
.slabel {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--t3);
  margin: 0 0 0.7rem;
}

/* ── Result panel ─────────────────────────────────── */
.result-wrap {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.4rem;
  margin-bottom: 0.8rem;
}
.result-wrap.danger  { border-color: var(--red);    background: linear-gradient(135deg, var(--red-dim) 0%, var(--bg2) 60%); }
.result-wrap.warning { border-color: var(--orange); background: linear-gradient(135deg, var(--orange-dim) 0%, var(--bg2) 60%); }
.result-wrap.safe    { border-color: var(--green);  background: linear-gradient(135deg, var(--green-dim) 0%, var(--bg2) 60%); }

.result-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--t3); margin-bottom: 6px; }
.result-name  { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; margin: 0 0 8px; line-height: 1.2; }
.result-name.danger  { color: var(--red); }
.result-name.warning { color: var(--orange); }
.result-name.safe    { color: var(--green); }

.badge {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 0.72rem; font-weight: 600;
  padding: 3px 10px; border-radius: 99px;
  border: 1px solid transparent;
}
.badge.danger  { background: var(--red-dim);    color: #ff8fa3; border-color: #5a1425; }
.badge.warning { background: var(--orange-dim); color: #ffb380; border-color: #5a3010; }
.badge.safe    { background: var(--green-dim);  color: #6ee7b7; border-color: #0d4a32; }

.conf-value {
  font-family: 'DM Mono', monospace;
  font-size: 2rem; font-weight: 500;
  color: var(--t1); line-height: 1;
}

/* ── Probability bars ─────────────────────────────── */
.pbar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
.pbar-name { font-size: 0.8rem; color: var(--t2); width: 185px; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pbar-track { flex: 1; height: 6px; background: var(--bg4); border-radius: 99px; overflow: hidden; }
.pbar-fill { height: 100%; border-radius: 99px; }
.pbar-fill.danger  { background: var(--red); }
.pbar-fill.warning { background: var(--orange); }
.pbar-fill.safe    { background: var(--green); }
.pbar-fill.accent  { background: var(--accent); }
.pbar-pct { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: var(--t3); width: 42px; text-align: right; flex-shrink: 0; }

/* ── Reco box ─────────────────────────────────────── */
.reco-box {
  background: var(--bg3);
  border: 1px solid var(--border2);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius-sm);
  padding: 0.9rem 1rem;
  margin-top: 0.8rem;
}
.reco-box p { font-size: 0.82rem; color: var(--t2); margin: 0; line-height: 1.75; }

/* ── Preview image card ───────────────────────────── */
.img-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 0.8rem;
}
.img-card-footer {
  padding: 8px 12px;
  font-size: 0.75rem;
  color: var(--t3);
  font-family: 'DM Mono', monospace;
  border-top: 1px solid var(--border);
  background: var(--bg3);
}

/* ── Divider ──────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Riwayat table ────────────────────────────────── */
.rtable-head {
  display: grid;
  grid-template-columns: 36px 130px 52px 1fr 100px 90px;
  gap: 0;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: var(--radius) var(--radius) 0 0;
  padding: 10px 14px;
}
.rtable-hcell {
  font-size: 0.65rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase; color: var(--t3);
}

/* ── Tentang cards ────────────────────────────────── */
.info-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.1rem 1.2rem;
  margin-bottom: 0.7rem;
  transition: border-color 0.2s;
}
.info-card:hover { border-color: var(--border2); }
.info-card-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--t3); margin-bottom: 4px; }
.info-card-value { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 600; color: var(--t1); }

.kelas-row {
  display: flex; align-items: center;
  justify-content: space-between;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.65rem 0.9rem;
  margin-bottom: 6px;
  transition: border-color 0.15s;
}
.kelas-row:hover { border-color: var(--border2); }
.kelas-name { font-size: 0.85rem; font-weight: 500; color: var(--t1); }
.kelas-code { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--t3); margin-left: 6px; }

/* ── Disclaimer strip ─────────────────────────────── */
.disc-strip {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.7rem 1rem;
  font-size: 0.78rem;
  color: var(--t3);
  line-height: 1.6;
  margin-top: 0.8rem;
}

/* ── Progress bar override ────────────────────────── */
.stProgress > div > div { background: var(--accent) !important; border-radius: 99px !important; }

/* ── Streamlit image ──────────────────────────────── */
div[data-testid="caption"] { font-size: 0.72rem !important; color: var(--t3) !important; }

/* ── Spinner ──────────────────────────────────────── */
div[data-testid="stSpinner"] p { color: var(--t2) !important; font-size: 0.85rem !important; }

/* ── stat chip ────────────────────────────────────── */
.stat-chip {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--bg3); border: 1px solid var(--border2);
  border-radius: 99px; padding: 4px 12px;
  font-size: 0.75rem; color: var(--t2);
  font-family: 'DM Mono', monospace;
}
.stat-chip b { color: var(--t1); font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# CLASS DATA
# ═════════════════════════════════════════════════════════════
CLASS_INFO = {
    "akiec": {
        "name": "Actinic Keratoses",
        "kategori": "Pra-kanker",
        "variant": "warning",
        "warna": "🟠",
        "rekomendasi": (
            "Lesi pra-kanker akibat paparan sinar UV berlebih. "
            "Segera periksakan ke dokter spesialis kulit untuk penanganan dini "
            "sebelum berkembang menjadi karsinoma sel skuamosa."
        ),
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "kategori": "Ganas (Kanker)",
        "variant": "danger",
        "warna": "🔴",
        "rekomendasi": (
            "Kanker kulit yang memerlukan penanganan medis segera. "
            "Meskipun jarang bermetastasis, BCC dapat merusak jaringan sekitar. "
            "Segera konsultasikan ke dermatologis."
        ),
    },
    "bkl": {
        "name": "Benign Keratosis",
        "kategori": "Jinak",
        "variant": "safe",
        "warna": "🟢",
        "rekomendasi": (
            "Lesi jinak meliputi seborrheic keratoses dan solar lentigo. "
            "Umumnya tidak berbahaya. Tetap disarankan pemeriksaan klinis "
            "untuk konfirmasi diagnosis."
        ),
    },
    "df": {
        "name": "Dermatofibroma",
        "kategori": "Jinak",
        "variant": "safe",
        "warna": "🟢",
        "rekomendasi": (
            "Benjolan jinak pada kulit. Umumnya tidak memerlukan pengobatan "
            "kecuali mengganggu aktivitas atau terdapat perubahan yang mencurigakan."
        ),
    },
    "mel": {
        "name": "Melanoma",
        "kategori": "Ganas (Kanker)",
        "variant": "danger",
        "warna": "🔴",
        "rekomendasi": (
            "Melanoma adalah kanker kulit paling berbahaya yang berasal dari sel melanosit. "
            "Deteksi dini sangat menentukan keberhasilan pengobatan. "
            "Segera periksakan ke dermatologis untuk tindak lanjut klinis."
        ),
    },
    "nv": {
        "name": "Melanocytic Nevi",
        "kategori": "Jinak",
        "variant": "safe",
        "warna": "🟢",
        "rekomendasi": (
            "Tahi lalat biasa yang bersifat jinak. Pantau secara berkala "
            "menggunakan kriteria ABCDE. Konsultasikan jika ada perubahan "
            "ukuran, warna, atau bentuk."
        ),
    },
    "vasc": {
        "name": "Vascular Lesions",
        "kategori": "Jinak",
        "variant": "safe",
        "warna": "🟢",
        "rekomendasi": (
            "Lesi pembuluh darah seperti hemangioma dan angioma. "
            "Umumnya jinak. Konsultasikan ke dokter jika membesar atau berdarah."
        ),
    },
}

# ═════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════
for key, val in {
    "page": "Deteksi Lesi Kulit",
    "riwayat": [],
    "hasil": None,
    "img_pil": None,
    "img_name": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ═════════════════════════════════════════════════════════════
# MODEL LOAD
# ═════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_assets():
    m = tf.keras.models.load_model("model_final.h5")
    with open("label_map.json") as f:
        lm = json.load(f)
    return m, lm

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def img_to_b64(img: Image.Image, size=(56, 56)) -> str:
    t = img.copy().convert("RGB")
    t.thumbnail(size)
    buf = io.BytesIO()
    t.save(buf, format="JPEG", quality=75)
    return base64.b64decode(base64.b64encode(buf.getvalue())).decode()

def img_b64_str(img: Image.Image, size=(56, 56)) -> str:
    t = img.copy().convert("RGB")
    t.thumbnail(size)
    buf = io.BytesIO()
    t.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()

try:
    model, label_map = load_assets()
    if all(isinstance(v, int) for v in label_map.values()):
        idx2label = {v: k for k, v in label_map.items()}
    else:
        idx2label = {int(k): v for k, v in label_map.items()}
    model_ok = True
except Exception as _e:
    model_ok = False
    _model_err = str(_e)

# ═════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="brand-block">
      <span class="brand-icon">🔬</span>
      <p class="brand-title">SPK Deteksi Dini<br>Kanker Kulit</p>
      <p class="brand-sub">MobileNetV2 · HAM10000 · Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="nav-label">Menu</p>', unsafe_allow_html=True)

    nav_items = [
        ("Deteksi Lesi Kulit", "🔍"),
        ("Riwayat Prediksi",   "📋"),
        ("Tentang Sistem",     "ℹ️"),
    ]
    for page, icon in nav_items:
        active = st.session_state.page == page
        wrap = st.container()
        if active:
            wrap.markdown('<div class="nav-active">', unsafe_allow_html=True)
        if wrap.button(f"{icon}  {page}", key=f"nav_{page}", use_container_width=True):
            st.session_state.page = page
            st.rerun()
        if active:
            wrap.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="nav-label">⚠ Disclaimer</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="disc-box">
      <p>Aplikasi ini adalah <strong>alat bantu skripsi</strong> dan
      <strong>bukan pengganti diagnosis medis profesional</strong>.
      Hasil prediksi wajib dikonfirmasi oleh dokter spesialis kulit.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p style="font-size:.68rem;color:#2e3550;text-align:center">'
        'v1.0 — Tugas Akhir IF 2025</p>',
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════════════════
# ── HALAMAN 1: DETEKSI LESI KULIT ────────────────────────────
# ═════════════════════════════════════════════════════════════
if st.session_state.page == "Deteksi Lesi Kulit":

    st.markdown('<h1 class="page-heading">Deteksi Lesi Kulit</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Unggah citra dermoskopi untuk dianalisis oleh model MobileNetV2 — '
        'hasil tampil langsung di halaman ini.</p>',
        unsafe_allow_html=True,
    )

    if not model_ok:
        st.error(f"❌ Gagal memuat model: {_model_err}")
        st.stop()

    # ── Two-column layout ──
    col_left, col_right = st.columns([1, 1.25], gap="large")

    # ── LEFT: upload + image preview ──
    with col_left:
        st.markdown('<p class="slabel">Unggah Citra</p>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "upload",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="upload-hint">Format: <span>JPG</span> <span>JPEG</span>'
            ' <span>PNG</span> · Maks. <span>5 MB</span></div>',
            unsafe_allow_html=True,
        )

        # Handle file
        if uploaded is not None:
            if uploaded.size > 5 * 1024 * 1024:
                st.error("❌ File melebihi 5 MB.")
            else:
                st.session_state.img_pil  = Image.open(uploaded)
                st.session_state.img_name = uploaded.name

        # Show preview
        if st.session_state.img_pil is not None:
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.image(st.session_state.img_pil, use_container_width=True)
            st.markdown(
                f'<div class="img-card-footer">📎 {st.session_state.img_name}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Buttons ──
        c1, c2 = st.columns(2, gap="small")
        with c1:
            run_btn = st.button(
                "Analisis Sekarang", type="primary",
                use_container_width=True, key="btn_run",
            )
        with c2:
            reset_btn = st.button(
                "Reset", type="secondary",
                use_container_width=True, key="btn_reset",
            )

        if reset_btn:
            st.session_state.img_pil  = None
            st.session_state.img_name = ""
            st.session_state.hasil    = None
            st.rerun()

        if run_btn:
            if st.session_state.img_pil is None:
                st.warning("⚠️ Unggah citra terlebih dahulu.")
            else:
                with st.spinner("Menganalisis citra…"):
                    arr  = preprocess(st.session_state.img_pil)
                    preds = model.predict(arr, verbose=0)[0]

                top_i   = int(np.argmax(preds))
                top_lbl = idx2label[top_i]
                top_conf = float(preds[top_i])
                info    = CLASS_INFO.get(top_lbl, {})

                all_probs = sorted(
                    [{"label": idx2label[i],
                      "name": CLASS_INFO.get(idx2label[i], {}).get("name", idx2label[i]),
                      "prob": float(preds[i]),
                      "variant": CLASS_INFO.get(idx2label[i], {}).get("variant", "safe")}
                     for i in range(len(preds))],
                    key=lambda x: x["prob"], reverse=True,
                )

                st.session_state.hasil = {
                    "label":    top_lbl,
                    "name":     info.get("name", top_lbl),
                    "kategori": info.get("kategori", "-"),
                    "variant":  info.get("variant", "safe"),
                    "conf":     top_conf,
                    "reko":     info.get("rekomendasi", ""),
                    "probs":    all_probs,
                    "waktu":    datetime.now().strftime("%d %b %Y, %H:%M"),
                    "img_b64":  img_b64_str(st.session_state.img_pil),
                    "img_name": st.session_state.img_name,
                }

                # Simpan ke riwayat
                st.session_state.riwayat.append({
                    "waktu":    st.session_state.hasil["waktu"],
                    "img_b64":  st.session_state.hasil["img_b64"],
                    "prediksi": f"{info.get('name', top_lbl)} ({top_lbl})",
                    "prob":     f"{top_conf*100:.1f}%",
                    "status":   info.get("kategori", "-"),
                    "variant":  info.get("variant", "safe"),
                })

                st.rerun()

        st.markdown(
            '<div class="disc-strip">Hasil analisis ini bersifat sebagai alat bantu skrining awal '
            '(second opinion) dan bukan pengganti diagnosis klinis oleh dokter spesialis kulit.</div>',
            unsafe_allow_html=True,
        )

    # ── RIGHT: hasil prediksi ──
    with col_right:
        if st.session_state.hasil is None:
            # Empty state
            st.markdown("""
            <div style="height:100%;min-height:380px;display:flex;flex-direction:column;
                        align-items:center;justify-content:center;text-align:center;
                        background:var(--bg2);border:1px dashed var(--border);
                        border-radius:var(--radius);padding:2.5rem">
              <div style="font-size:2.5rem;margin-bottom:1rem;opacity:.4">🩺</div>
              <p style="font-size:.9rem;color:var(--t3);max-width:220px;line-height:1.6;margin:0">
                Unggah citra dan klik <strong style="color:var(--t2)">Analisis Sekarang</strong>
                untuk melihat hasil prediksi di sini
              </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            h = st.session_state.hasil
            v = h["variant"]

            # ── Prediksi utama ──
            st.markdown('<p class="slabel">Hasil Prediksi</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-wrap {v}">
              <div class="result-label">Prediksi Utama</div>
              <div class="result-name {v}">{h['name']}</div>
              <span class="badge {v}">
                {'⚠' if v=='warning' else ('✕' if v=='danger' else '✓')} {h['kategori']}
              </span>
            </div>
            """, unsafe_allow_html=True)

            # ── Confidence ──
            st.markdown(f"""
            <div class="result-wrap" style="padding:1rem 1.4rem">
              <div class="result-label">Tingkat Keyakinan</div>
              <div class="conf-value">{h['conf']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Probability bars ──
            st.markdown('<p class="slabel" style="margin-top:.5rem">Distribusi Probabilitas</p>', unsafe_allow_html=True)
            max_p = h["probs"][0]["prob"] if h["probs"] else 1
            bars_html = ""
            for item in h["probs"]:
                width = (item["prob"] / max_p * 100) if max_p > 0 else 0
                bars_html += f"""
                <div class="pbar-row">
                  <div class="pbar-name">{item['name']} <span style="color:var(--t3);font-size:.7rem">({item['label']})</span></div>
                  <div class="pbar-track">
                    <div class="pbar-fill {item['variant']}" style="width:{width:.1f}%"></div>
                  </div>
                  <div class="pbar-pct">{item['prob']*100:.1f}%</div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

            # ── Rekomendasi ──
            st.markdown(f"""
            <div class="reco-box">
              <div class="result-label" style="margin-bottom:5px">Rekomendasi Medis</div>
              <p>{h['reko']}</p>
            </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# ── HALAMAN 2: RIWAYAT PREDIKSI ──────────────────────────────
# ═════════════════════════════════════════════════════════════
elif st.session_state.page == "Riwayat Prediksi":

    st.markdown('<h1 class="page-heading">Riwayat Prediksi</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Seluruh sesi pemeriksaan yang telah dilakukan.</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.riwayat:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;background:var(--bg2);
                    border:1px dashed var(--border);border-radius:var(--radius)">
          <div style="font-size:2rem;opacity:.35;margin-bottom:1rem">📋</div>
          <p style="color:var(--t3);font-size:.88rem;margin:0">
            Belum ada riwayat. Mulai deteksi di menu Deteksi Lesi Kulit.
          </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Mulai Deteksi", type="primary", key="go_deteksi"):
            st.session_state.page = "Deteksi Lesi Kulit"
            st.rerun()
    else:
        n = len(st.session_state.riwayat)
        st.markdown(
            f'<p class="stat-chip"><b>{n}</b> pemeriksaan tercatat</p>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Header ──
        st.markdown("""
        <div class="rtable-head">
          <span class="rtable-hcell">No</span>
          <span class="rtable-hcell">Waktu</span>
          <span class="rtable-hcell">Citra</span>
          <span class="rtable-hcell">Prediksi</span>
          <span class="rtable-hcell">Prob.</span>
          <span class="rtable-hcell">Status</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Rows — native Streamlit columns ──
        rev = list(reversed(st.session_state.riwayat))
        for i, r in enumerate(rev, 1):
            is_last = i == len(rev)
            br = f"0 0 {12}px {12}px" if is_last else "0"
            bg = "#111420" if i % 2 == 0 else "#0f1220"
            s = (f"background:{bg};border-left:1px solid #252a40;"
                 f"border-right:1px solid #252a40;border-bottom:1px solid #252a40;"
                 f"border-radius:{br};padding:9px 5px;")

            cno, cwk, cim, cpd, cpr, cst = st.columns([0.36, 1.3, 0.52, 2.0, 0.95, 0.9])

            with cno:
                st.markdown(
                    f'<div style="{s}padding-left:14px;color:#4a5270;'
                    f'font-size:.82rem;line-height:46px">{i}</div>',
                    unsafe_allow_html=True)
            with cwk:
                st.markdown(
                    f'<div style="{s}color:#8892b0;font-size:.78rem;'
                    f'padding-top:9px;line-height:1.45">{r["waktu"]}</div>',
                    unsafe_allow_html=True)
            with cim:
                st.markdown(f'<div style="{s}">', unsafe_allow_html=True)
                try:
                    img_bytes = base64.b64decode(r["img_b64"])
                    thumb = Image.open(io.BytesIO(img_bytes))
                    st.image(thumb, width=44)
                except Exception:
                    st.markdown("—")
                st.markdown("</div>", unsafe_allow_html=True)
            with cpd:
                st.markdown(
                    f'<div style="{s}color:#eef0f8;font-size:.82rem;'
                    f'padding-top:9px;line-height:1.45">{r["prediksi"]}</div>',
                    unsafe_allow_html=True)
            with cpr:
                st.markdown(
                    f'<div style="{s}font-family:\'DM Mono\',monospace;'
                    f'color:#eef0f8;font-size:.82rem;line-height:46px;text-align:center">'
                    f'{r["prob"]}</div>',
                    unsafe_allow_html=True)
            with cst:
                v = r.get("variant", "safe")
                st.markdown(
                    f'<div style="{s}padding-top:10px;text-align:center">'
                    f'<span class="badge {v}">{r["status"]}</span></div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ca, cb, _ = st.columns([1, 1, 2])
        with ca:
            if st.button("Hapus Semua", type="secondary", use_container_width=True, key="hapus"):
                st.session_state.riwayat = []
                st.rerun()
        with cb:
            if st.button("Deteksi Baru", type="primary", use_container_width=True, key="baru"):
                st.session_state.hasil   = None
                st.session_state.img_pil = None
                st.session_state.page    = "Deteksi Lesi Kulit"
                st.rerun()

# ═════════════════════════════════════════════════════════════
# ── HALAMAN 3: TENTANG SISTEM ─────────────────────────────────
# ═════════════════════════════════════════════════════════════
elif st.session_state.page == "Tentang Sistem":

    st.markdown('<h1 class="page-heading">Tentang Sistem</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Informasi teknis, arsitektur model, dan kelas lesi kulit yang didukung.</p>',
        unsafe_allow_html=True,
    )

    col_info, col_kelas = st.columns([1, 1.1], gap="large")

    with col_info:
        st.markdown('<p class="slabel">Informasi Teknis</p>', unsafe_allow_html=True)
        specs = [
            ("Nama Sistem",        "SPK Deteksi Dini Kanker Kulit"),
            ("Arsitektur Model",   "MobileNetV2 + Transfer Learning"),
            ("Dataset",            "HAM10000 — 10.015 citra"),
            ("Jumlah Kelas",       "7 kelas lesi kulit"),
            ("Framework",          "TensorFlow + Streamlit"),
            ("Input Citra",        "224 × 224 piksel (RGB)"),
            ("Platform Deploy",    "Streamlit Community Cloud"),
        ]
        for label, val in specs:
            st.markdown(f"""
            <div class="info-card">
              <div class="info-card-label">{label}</div>
              <div class="info-card-value">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_kelas:
        st.markdown('<p class="slabel">7 Kelas Lesi Kulit</p>', unsafe_allow_html=True)
        for code, info in CLASS_INFO.items():
            v = info["variant"]
            st.markdown(f"""
            <div class="kelas-row">
              <div>
                <span class="kelas-name">{info['warna']} {info['name']}</span>
                <span class="kelas-code">({code})</span>
              </div>
              <span class="badge {v}">{info['kategori']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="slabel">Cara Penggunaan</p>', unsafe_allow_html=True)
        steps = [
            "Buka halaman Deteksi Lesi Kulit",
            "Unggah citra dermoskopi (JPG/PNG, maks. 5 MB)",
            "Klik tombol Analisis Sekarang",
            "Baca hasil prediksi dan distribusi probabilitas",
            "Konsultasikan hasil ke dokter spesialis kulit",
        ]
        for i, s in enumerate(steps, 1):
            st.markdown(f"""
            <div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:8px">
              <div style="flex-shrink:0;width:22px;height:22px;border-radius:50%;
                          background:var(--accent-dim);border:1px solid var(--accent);
                          font-size:.72rem;font-weight:700;color:#a8c0ff;
                          display:flex;align-items:center;justify-content:center">{i}</div>
              <p style="font-size:.83rem;color:var(--t2);margin:2px 0 0;line-height:1.5">{s}</p>
            </div>
            """, unsafe_allow_html=True)
