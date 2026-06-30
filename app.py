import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import tempfile
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import datetime

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="MMBI Engine - Interest Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0a0e1a 0%, #0e1117 100%);
    }
    .hero-title {
        font-size: 52px;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-top: 20px;
        margin-bottom: 5px;
    }
    .hero-subtitle {
        font-size: 19px;
        color: #9ca3af;
        text-align: center;
        margin-bottom: 10px;
    }
    .hero-tagline {
        font-size: 15px;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 500;
    }
    .feature-card {
        background: #161a2b;
        border: 1px solid #262b3d;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        height: 100%;
    }
    .feature-icon { font-size: 32px; margin-bottom: 10px; }
    .feature-title {
        font-size: 16px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }
    .feature-desc {
        font-size: 13px;
        color: #9ca3af;
        line-height: 1.5;
    }
    .verdict-interested {
        background: linear-gradient(135deg, #0a2e0a, #0f3d0f);
        border: 2px solid #00aa00;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        color: #00ff88;
        font-size: 22px;
        font-weight: 800;
    }
    .verdict-neutral {
        background: linear-gradient(135deg, #2e2a0a, #3d370f);
        border: 2px solid #aaaa00;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        color: #ffd700;
        font-size: 22px;
        font-weight: 800;
    }
    .verdict-not {
        background: linear-gradient(135deg, #2e0a0a, #3d0f0f);
        border: 2px solid #aa0000;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        color: #ff5566;
        font-size: 22px;
        font-weight: 800;
    }
    .footer-text {
        text-align: center;
        color: #555;
        font-size: 13px;
        padding: 30px 0 10px 0;
    }
    section[data-testid="stSidebar"] {
        background: #0c0f1a;
    }
</style>
""", unsafe_allow_html=True)

# ── HERO SECTION ──────────────────────────────────────────
st.markdown('<div class="hero-title">🎯 MMBI Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">AI-Powered Viewer Interest '
    'Analysis for Product Review Videos</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="hero-tagline">Upload a video → Get instant '
    'interest analytics powered by computer vision</div>',
    unsafe_allow_html=True
)

# ── FEATURE STRIP ─────────────────────────────────────────
f1, f2, f3, f4 = st.columns(4)
features = [
    ("🧠", "Emotion AI",
     "Deep learning model trained on facial expressions"),
    ("👥", "Multi-Person",
     "Tracks up to 4 viewers in one video simultaneously"),
    ("📊", "Live Timeline",
     "See exactly when interest peaked or dropped"),
    ("📄", "PDF Reports",
     "Professional downloadable analysis for clients"),
]
for col, (icon, title, desc) in zip(
        [f1, f2, f3, f4], features):
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ── LOAD MODEL ────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.h5")
    with open("class_names.json") as f:
        class_names = json.load(f)
    face_det = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )
    return model, class_names, face_det

INTEREST_MAP = {
    "happy": 1.0, "surprise": 1.0, "neutral": 0.50,
    "fear": 0.40, "sad": 0.25, "angry": 0.1, "disgust": 0.1,
}

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    max_people = st.slider("Max people to track", 1, 4, 2)
    scan_secs  = st.slider("Scan duration (seconds)", 5, 30, 10)
    threshold  = st.slider("Detection sensitivity", 0.3, 0.7, 0.4)
    st.markdown("---")
    st.markdown("### ℹ️ About MMBI Engine")
    st.markdown(
        "MMBI Engine analyzes facial expressions to "
        "predict viewer interest in product review "
        "videos — helping brands understand real "
        "audience engagement."
    )
    st.markdown("---")
    st.caption("Built with TensorFlow + OpenCV")

# ── MAIN UPLOAD ───────────────────────────────────────────
st.markdown("### 📤 Upload Your Video")
uploaded = st.file_uploader(
    "Drop a product review video here",
    type=["mp4", "avi", "mov"],
    label_visibility="collapsed"
)

if uploaded is not None:
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4")
    tmp.write(uploaded.read())
    tmp.close()
    video_path = tmp.name

    st.video(uploaded)
    st.markdown("---")

    if st.button("🚀 Analyze Video",
                 use_container_width=True, type="primary"):

        with st.spinner("Loading AI model..."):
            model, class_names, face_det = load_model()

        progress = st.progress(0)
        status   = st.empty()

        status.text("🔍 Finding people in video...")
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        people= {}

        for i in range(int(fps * scan_secs)):
            ret, frame = cap.read()
            if not ret: break
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_det.detectMultiScale(
                gray, 1.1, 10, minSize=(100, 100))
            for (x, y, w, h) in faces:
                gray_roi  = gray[y:y+h, x:x+w].copy()
                face_size = w * h
                matched_id, best_sim = None, threshold
                for pid, pdata in people.items():
                    curr_h = cv2.calcHist(
                        [gray_roi], [0], None, [256], [0, 256])
                    cv2.normalize(curr_h, curr_h)
                    sim = cv2.compareHist(
                        pdata["hist"], curr_h, cv2.HISTCMP_CORREL)
                    if sim > best_sim:
                        best_sim, matched_id = sim, pid
                if matched_id is not None:
                    if face_size > people[matched_id]["size"]:
                        h_roi = cv2.calcHist(
                            [gray_roi], [0], None, [256], [0, 256])
                        cv2.normalize(h_roi, h_roi)
                        people[matched_id] = {
                            "face": (x, y, w, h),
                            "frame": frame.copy(),
                            "size": face_size, "hist": h_roi}
                else:
                    if len(people) < max_people:
                        pid = len(people) + 1
                        h_roi = cv2.calcHist(
                            [gray_roi], [0], None, [256], [0, 256])
                        cv2.normalize(h_roi, h_roi)
                        people[pid] = {
                            "face": (x, y, w, h),
                            "frame": frame.copy(),
                            "size": face_size, "hist": h_roi}
        cap.release()

        st.markdown(f"### 👥 Found {len(people)} Person(s)")
        if people:
            cols = st.columns(len(people))
            for pid, pdata in people.items():
                x, y, w, h = pdata["face"]
                fimg = cv2.cvtColor(
                    pdata["frame"][y:y+h, x:x+w],
                    cv2.COLOR_BGR2RGB)
                cols[pid-1].image(
                    fimg, caption=f"Person {pid}",
                    use_column_width=True)

        status.text("🎬 Analyzing video frame by frame...")
        timelines = {pid: [] for pid in people.keys()}
        cap, fcount = cv2.VideoCapture(video_path), 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            ts   = fcount / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if fcount % 2 == 0:
                faces = face_det.detectMultiScale(
                    gray, 1.3, 5, minSize=(40, 40))
                for (x, y, w, h) in faces:
                    gray_roi = gray[y:y+h, x:x+w]
                    curr_h   = cv2.calcHist(
                        [gray_roi], [0], None, [256], [0, 256])
                    cv2.normalize(curr_h, curr_h)
                    best_pid, best_sim = None, 0.35
                    for pid, pdata in people.items():
                        sim = cv2.compareHist(
                            pdata["hist"], curr_h,
                            cv2.HISTCMP_CORREL)
                        if sim > best_sim:
                            best_sim, best_pid = sim, pid
                    if best_pid is None:
                        continue
                    roi_in = cv2.resize(
                        gray_roi, (48, 48)
                    ).astype("float32") / 255.0
                    roi_in = np.expand_dims(roi_in, axis=(0, -1))
                    preds  = model.predict(roi_in, verbose=0)[0]
                    emo_i  = int(np.argmax(preds))
                    emotion= class_names[emo_i]
                    conf   = float(preds[emo_i])
                    if conf < 0.55:
                        emotion = "neutral"
                    score = round(
                        INTEREST_MAP.get(emotion, 0.5) * 100, 1)
                    timelines[best_pid].append(
                        {"time": round(ts, 2), "score": score})
            progress.progress(min(int(fcount/total*100), 100))
            fcount += 1
        cap.release()
        status.text("✅ Analysis complete!")
        progress.progress(100)

        st.markdown("---")
        st.markdown("## 📊 Results")

        for pid, tl in timelines.items():
            if not tl:
                continue
            st.markdown(f"### Person {pid}")
            df = pd.DataFrame(tl)
            df["sm"] = df["score"].rolling(
                30, min_periods=1).mean()
            scores = df["score"].values
            avg    = np.mean(scores)
            inter  = np.mean(scores >= 65) * 100
            neut   = np.mean(
                (scores >= 40) & (scores < 65)) * 100
            not_i  = np.mean(scores < 40) * 100

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Score", f"{avg:.1f}/100")
            c2.metric("Interested", f"{inter:.1f}%")
            c3.metric("Neutral", f"{neut:.1f}%")
            c4.metric("Not Interested", f"{not_i:.1f}%")

            if avg >= 65:
                st.markdown(
                    '<div class="verdict-interested">'
                    '✅ HIGHLY INTERESTED</div>',
                    unsafe_allow_html=True)
            elif avg >= 40:
                st.markdown(
                    '<div class="verdict-neutral">'
                    '➡️ NEUTRAL</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="verdict-not">'
                    '❌ NOT INTERESTED</div>',
                    unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(12, 4))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#161a2b")
            t, s = df["time"].values, df["sm"].values
            ax.fill_between(t, 65, 100, alpha=0.12, color="green")
            ax.fill_between(t, 40, 65, alpha=0.12, color="yellow")
            ax.fill_between(t, 0, 40, alpha=0.12, color="red")
            ax.plot(t, s, color="#00d4ff", lw=2.5)
            ax.axhline(65, color="green", lw=0.8, ls="--")
            ax.axhline(40, color="orange", lw=0.8, ls="--")
            ax.set_xlim(0, t[-1]); ax.set_ylim(0, 100)
            ax.set_xlabel("Time (s)", color="white")
            ax.set_ylabel("Interest Score", color="white")
            ax.set_title(f"Person {pid} Interest Timeline",
                         color="white")
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333")
            st.pyplot(fig)
            plt.close()
            st.markdown("---")

        os.unlink(video_path)
        st.success("✅ Analysis complete!")

# ── FOOTER ────────────────────────────────────────────────
st.markdown(
    '<div class="footer-text">MMBI Engine © 2026 · '
    'Powered by TensorFlow & OpenCV · '
    'Built for product review analytics</div>',
    unsafe_allow_html=True
)
