import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
import tempfile
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import datetime

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="MMBI Engine — Know Your Audience",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
#  CUSTOM CSS — BRANDING
# ============================================================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .hero {
        text-align: center; padding: 40px 20px;
        background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 100%);
        border-radius: 15px; margin-bottom: 30px; border: 1px solid #2a2a3e;
    }
    .hero-title {
        font-size: 48px; font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .hero-subtitle { font-size: 18px; color: #aaaaaa; max-width: 600px; margin: 0 auto; line-height: 1.6; }
    .feature-box { background: #1e2130; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #2a2a3e; height: 100%; }
    .feature-icon { font-size: 32px; margin-bottom: 10px; }
    .feature-title { font-size: 16px; font-weight: 600; color: #00d4ff; margin-bottom: 8px; }
    .feature-desc { font-size: 13px; color: #888; }
    .metric-value { font-size: 36px; font-weight: bold; color: #00d4ff; }
    .verdict-interested { background: linear-gradient(135deg, #0a2e0a, #0d3d0d); border: 2px solid #00aa00; border-radius: 12px; padding: 18px; text-align: center; color: #00ff88; font-size: 22px; font-weight: bold; }
    .verdict-neutral { background: linear-gradient(135deg, #2e2a0a, #3d370d); border: 2px solid #aaaa00; border-radius: 12px; padding: 18px; text-align: center; color: #ffdd00; font-size: 22px; font-weight: bold; }
    .verdict-not { background: linear-gradient(135deg, #2e0a0a, #3d0d0d); border: 2px solid #aa0000; border-radius: 12px; padding: 18px; text-align: center; color: #ff5555; font-size: 22px; font-weight: bold; }
    .footer { text-align: center; padding: 30px; color: #555; font-size: 13px; border-top: 1px solid #2a2a3e; margin-top: 40px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-title">🎯 MMBI Engine</div>
    <div class="hero-subtitle">
        Upload any product review or reaction video.
        Our AI analyzes facial expressions and body language
        to tell you exactly how interested your viewers really are —
        second by second, with visual explanations for every call.
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
for col, icon, title, desc in [
    (col1, "🎭", "Emotion Detection", "MobileNetV2, 65% test accuracy on FER-2013"),
    (col2, "🔥", "Grad-CAM Explainability", "See exactly what the AI is looking at"),
    (col3, "📊", "Interest Timeline", "See exactly when interest peaks or drops"),
    (col4, "📄", "PDF Reports", "Professional reports ready to share"),
]:
    with col:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================
#  LOAD MODEL — MobileNetV2, 96x96 RGB
# ============================================================
IMG_SIZE = (96, 96)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.h5")
    with open("class_names.json") as f:
        class_names = json.load(f)
    face_det = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return model, class_names, face_det

INTEREST_MAP = {
    "happy": 1.0, "surprise": 1.0, "neutral": 0.75,
    "fear": 0.5, "sad": 0.25, "angry": 0.1, "disgust": 0.1,
}
PERSON_COLORS_BGR = [(0,200,255), (255,100,0), (0,255,100), (255,0,200)]
PERSON_COLORS_PLT = ["#00c8ff", "#ff6400", "#00ff64", "#ff00c8"]


def predict_emotion(model, class_names, face_roi_bgr):
    roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, IMG_SIZE)
    roi_in = preprocess_input(roi_resized.astype("float32"))
    roi_in = np.expand_dims(roi_in, axis=0)
    preds = model.predict(roi_in, verbose=0)[0]
    idx = int(np.argmax(preds))
    return class_names[idx], float(preds[idx]), roi_resized


# ============================================================
#  GRAD-CAM — explains WHY the model predicted an emotion
#  by showing which pixels of the face influenced it most.
# ============================================================
def find_backbone_and_last_conv(model):
    """Locate the nested MobileNetV2 sub-model and its last conv layer."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for candidate in ["out_relu", "Conv_1"]:
                try:
                    layer.get_layer(candidate)
                    return layer, candidate
                except ValueError:
                    continue
    return None, None


@st.cache_resource
def build_gradcam_model(_model):
    backbone, conv_name = find_backbone_and_last_conv(_model)
    if backbone is None:
        return None
    grad_model = tf.keras.models.Model(
        inputs=_model.inputs,
        outputs=[backbone.get_layer(conv_name).output, _model.output],
    )
    return grad_model


def make_gradcam_heatmap(grad_model, img_array, pred_index):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(face_rgb_96, heatmap, alpha=0.45):
    heatmap_resized = cv2.resize(heatmap, (face_rgb_96.shape[1], face_rgb_96.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_uint8)[:, :, :3]
    heatmap_colored = np.uint8(heatmap_colored * 255)
    overlaid = np.uint8(face_rgb_96 * (1 - alpha) + heatmap_colored * alpha)
    return overlaid


# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    max_people = st.slider("Max people to track", 1, 4, 2)
    scan_secs  = st.slider("Scan duration (seconds)", 5, 30, 10)
    threshold  = st.slider("Detection sensitivity", 0.3, 0.7, 0.4)
    show_gradcam = st.checkbox("🔥 Show Grad-CAM explanations", value=True,
                                help="Highlights which parts of each face the AI focused on for its top prediction")

    st.markdown("---")
    st.markdown("## ℹ️ How it works")
    st.markdown("""
    1. Upload your video
    2. AI finds each person's face
    3. Tracks emotions frame by frame (MobileNetV2)
    4. Generates interest score over time
    5. Grad-CAM shows what drove each call
    6. Download report as PDF
    """)

    st.markdown("---")
    st.markdown("## 📈 Model Benchmark")
    st.markdown("""
    **Test accuracy: 65.0%** on FER-2013 (7-class)
    - Happy: 88% precision
    - Surprise: 75% precision
    - Fear / Sad: weakest classes (dataset-wide known issue)
    """)

    st.markdown("---")
    st.markdown("## 📧 Contact")
    st.markdown("MMBI Engine — reach out for partnership or API access.")

# ============================================================
#  MAIN UPLOAD
# ============================================================
uploaded = st.file_uploader(
    "Upload your product review video",
    type=["mp4", "avi", "mov"],
)

if uploaded is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read())
    tmp.close()
    video_path = tmp.name

    col_video, col_info = st.columns([2, 1])
    with col_video:
        st.video(uploaded)
    with col_info:
        st.markdown("### Ready to analyze")
        st.markdown(f"**File:** {uploaded.name}")
        st.markdown(f"**Size:** {uploaded.size / 1024 / 1024:.1f} MB")
        analyze_clicked = st.button("🚀 Analyze Video", use_container_width=True, type="primary")

    if uploaded is not None and 'analyze_clicked' in dir() and analyze_clicked:
        with st.spinner("Loading AI model..."):
            model, class_names, face_det = load_model()
            grad_model = build_gradcam_model(model) if show_gradcam else None

        progress = st.progress(0)
        status   = st.empty()

        # ── Find reference faces ─────────────────────────
        status.text("🔍 Finding people in video...")
        cap    = cv2.VideoCapture(video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        people = {}

        for i in range(int(fps * scan_secs)):
            ret, frame = cap.read()
            if not ret: break
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_det.detectMultiScale(gray, 1.1, 10, minSize=(100,100))
            for (x,y,w,h) in faces:
                gray_roi  = gray[y:y+h, x:x+w].copy()
                face_size = w * h
                matched_id, best_sim = None, threshold
                for pid, pdata in people.items():
                    curr_h = cv2.calcHist([gray_roi],[0],None,[256],[0,256])
                    cv2.normalize(curr_h, curr_h)
                    sim = cv2.compareHist(pdata["hist"], curr_h, cv2.HISTCMP_CORREL)
                    if sim > best_sim:
                        best_sim, matched_id = sim, pid
                if matched_id is not None:
                    if face_size > people[matched_id]["size"]:
                        h_roi = cv2.calcHist([gray_roi],[0],None,[256],[0,256])
                        cv2.normalize(h_roi, h_roi)
                        people[matched_id] = {"face": (x,y,w,h), "gray_roi": gray_roi,
                                               "frame": frame.copy(), "size": face_size, "hist": h_roi}
                else:
                    if len(people) < max_people:
                        pid = len(people) + 1
                        h_roi = cv2.calcHist([gray_roi],[0],None,[256],[0,256])
                        cv2.normalize(h_roi, h_roi)
                        people[pid] = {"face": (x,y,w,h), "gray_roi": gray_roi,
                                        "frame": frame.copy(), "size": face_size, "hist": h_roi}
        cap.release()

        st.markdown(f"### 👥 Found {len(people)} people")
        if people:
            cols = st.columns(len(people))
            for pid, pdata in people.items():
                x,y,w,h = pdata["face"]
                fimg = cv2.cvtColor(pdata["frame"][y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                cols[pid-1].image(fimg, caption=f"Person {pid}", use_container_width=True)

        # ── Analyze video ─────────────────────────────────
        status.text("🎬 Analyzing video frame by frame...")
        timelines = {pid: [] for pid in people.keys()}
        best_frame_per_person = {pid: {"conf": 0.0} for pid in people.keys()}
        emotion_counts = {pid: {} for pid in people.keys()}
        cap, fcount = cv2.VideoCapture(video_path), 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            ts   = fcount / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if fcount % 2 == 0:
                faces = face_det.detectMultiScale(gray, 1.3, 5, minSize=(40,40))
                for (x,y,w,h) in faces:
                    gray_roi = gray[y:y+h, x:x+w]
                    curr_h   = cv2.calcHist([gray_roi],[0],None,[256],[0,256])
                    cv2.normalize(curr_h, curr_h)
                    best_pid, best_sim = None, 0.35
                    for pid, pdata in people.items():
                        sim = cv2.compareHist(pdata["hist"], curr_h, cv2.HISTCMP_CORREL)
                        if sim > best_sim:
                            best_sim, best_pid = sim, pid
                    if best_pid is None: continue

                    face_roi_bgr = frame[y:y+h, x:x+w]
                    emotion, conf, face_rgb_96 = predict_emotion(model, class_names, face_roi_bgr)
                    if conf < 0.45: emotion = "neutral"
                    ew    = INTEREST_MAP.get(emotion, 0.5)
                    score = round(ew*100, 1)
                    timelines[best_pid].append({"time": round(ts,2), "score": score, "emotion": emotion})
                    emotion_counts[best_pid][emotion] = emotion_counts[best_pid].get(emotion, 0) + 1

                    if conf > best_frame_per_person[best_pid]["conf"]:
                        best_frame_per_person[best_pid] = {
                            "conf": conf, "emotion": emotion,
                            "face_rgb_96": face_rgb_96,
                            "pred_index": class_names.index(emotion) if emotion in class_names else 0,
                        }

            pct = int(fcount/total*100)
            progress.progress(min(pct,100))
            fcount += 1

        cap.release()
        status.text("✅ Analysis complete!")
        progress.progress(100)

        # ── Show results ───────────────────────────────────
        st.markdown("---")
        st.markdown("## 📊 Results")

        pdf_data = {}
        for pid, tl in timelines.items():
            if not tl: continue
            st.markdown(f"### Person {pid}")
            df = pd.DataFrame(tl)
            df["sm"] = df["score"].rolling(30, min_periods=1).mean()
            scores = df["score"].values
            avg   = np.mean(scores)
            inter = np.mean(scores>=65)*100
            neut  = np.mean((scores>=40)&(scores<65))*100
            not_i = np.mean(scores<40)*100
            pdf_data[pid] = dict(avg=avg, inter=inter, neut=neut, not_i=not_i,
                                  peak=max(scores), peak_t=tl[int(np.argmax(scores))]["time"])

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Avg Score", f"{avg:.1f}/100")
            c2.metric("Interested", f"{inter:.1f}%")
            c3.metric("Neutral", f"{neut:.1f}%")
            c4.metric("Not Interested", f"{not_i:.1f}%")

            if avg >= 65:
                st.markdown('<div class="verdict-interested">✅ HIGHLY INTERESTED</div>', unsafe_allow_html=True)
            elif avg >= 40:
                st.markdown('<div class="verdict-neutral">➡️ NEUTRAL</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict-not">❌ NOT INTERESTED</div>', unsafe_allow_html=True)

            # ── Timeline chart ──
            fig, ax = plt.subplots(figsize=(12,4))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#16213e")
            t, s = df["time"].values, df["sm"].values
            ax.fill_between(t,65,100,alpha=0.12,color="green")
            ax.fill_between(t,40,65,alpha=0.12,color="yellow")
            ax.fill_between(t,0,40,alpha=0.12,color="red")
            ax.plot(t,s,color=PERSON_COLORS_PLT[pid-1],lw=2.5)
            ax.axhline(65,color="green",lw=0.8,ls="--")
            ax.axhline(40,color="orange",lw=0.8,ls="--")
            ax.set_xlim(0,t[-1]); ax.set_ylim(0,100)
            ax.set_xlabel("Time(s)",color="white")
            ax.set_ylabel("Interest Score",color="white")
            ax.set_title(f"Person {pid} Interest Timeline",color="white")
            ax.tick_params(colors="white")
            for sp in ax.spines.values(): sp.set_edgecolor("#555")
            st.pyplot(fig)
            plt.close()

            # ── Emotion distribution + Grad-CAM side by side ──
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Emotion distribution**")
                counts = emotion_counts[pid]
                if counts:
                    fig2, ax2 = plt.subplots(figsize=(5,4))
                    fig2.patch.set_facecolor("#1a1a2e")
                    ax2.set_facecolor("#1a1a2e")
                    labels = list(counts.keys())
                    values = list(counts.values())
                    bar_colors = ["#00ff88" if INTEREST_MAP.get(l,0.5) >= 0.65 else
                                  "#ffdd00" if INTEREST_MAP.get(l,0.5) >= 0.4 else "#ff5555" for l in labels]
                    ax2.bar(labels, values, color=bar_colors)
                    ax2.set_ylabel("Frames", color="white")
                    ax2.tick_params(colors="white", rotation=30)
                    for sp in ax2.spines.values(): sp.set_edgecolor("#555")
                    st.pyplot(fig2)
                    plt.close()

            with rc2:
                st.markdown("**🔥 Grad-CAM — what the AI focused on**")
                if show_gradcam and grad_model is not None and best_frame_per_person[pid]["conf"] > 0:
                    bf = best_frame_per_person[pid]
                    img_in = preprocess_input(bf["face_rgb_96"].astype("float32"))
                    img_in = np.expand_dims(img_in, axis=0)
                    try:
                        heatmap = make_gradcam_heatmap(grad_model, img_in, bf["pred_index"])
                        overlaid = overlay_heatmap(bf["face_rgb_96"], heatmap)
                        cap_txt = f"Highest-confidence frame: '{bf['emotion']}' ({bf['conf']*100:.0f}%)"
                        st.image(overlaid, caption=cap_txt, use_container_width=True)
                    except Exception as e:
                        st.info(f"Grad-CAM unavailable for this model architecture ({e})")
                else:
                    st.info("Enable 'Show Grad-CAM explanations' in the sidebar to see this.")

            # ── CSV export of raw timeline ──
            csv_data = df[["time","score","emotion"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                f"⬇️ Download Person {pid} raw data (CSV)",
                csv_data, file_name=f"person_{pid}_timeline.csv", mime="text/csv"
            )
            st.markdown("---")

        # ── Cross-person comparison chart (only if 2+ people) ──
        if len([p for p in pdf_data]) > 1:
            st.markdown("## 🆚 Comparison Across People")
            fig3, ax3 = plt.subplots(figsize=(8,4))
            fig3.patch.set_facecolor("#1a1a2e")
            ax3.set_facecolor("#16213e")
            pids = list(pdf_data.keys())
            avgs = [pdf_data[p]["avg"] for p in pids]
            bar_colors = [PERSON_COLORS_PLT[p-1] for p in pids]
            ax3.bar([f"Person {p}" for p in pids], avgs, color=bar_colors)
            ax3.axhline(65, color="green", lw=0.8, ls="--")
            ax3.axhline(40, color="orange", lw=0.8, ls="--")
            ax3.set_ylabel("Avg Interest Score", color="white")
            ax3.set_ylim(0,100)
            ax3.tick_params(colors="white")
            for sp in ax3.spines.values(): sp.set_edgecolor("#555")
            st.pyplot(fig3)
            plt.close()

        # ── PDF report ──────────────────────────────────────
        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        c = canvas.Canvas(pdf_path, pagesize=A4)
        W, H = A4
        c.setFillColorRGB(0.05,0.05,0.3)
        c.rect(0,H-80,W,80,fill=1,stroke=0)
        c.setFont("Helvetica-Bold",22)
        c.setFillColor(colors.white)
        c.drawString(40,H-45,"MMBI Engine Report")
        c.setFont("Helvetica",11)
        c.drawString(40,H-65, f"Video: {uploaded.name}  |  Date: {datetime.datetime.now().strftime('%d %b %Y %H:%M')}")
        y = H-110
        for pid, d in pdf_data.items():
            c.setFillColorRGB(0.1,0.1,0.5)
            c.rect(30,y-5,W-60,30,fill=1,stroke=0)
            c.setFont("Helvetica-Bold",14)
            c.setFillColor(colors.white)
            c.drawString(40,y+8,f"Person {pid}")
            y -= 45
            c.setFont("Helvetica",11)
            c.setFillColor(colors.black)
            for label, val in [
                ("Average Score", f"{d['avg']:.1f}/100"),
                ("Interested", f"{d['inter']:.1f}%"),
                ("Neutral", f"{d['neut']:.1f}%"),
                ("Not Interested", f"{d['not_i']:.1f}%"),
                ("Peak Interest", f"{d['peak']:.0f}% at {d['peak_t']:.1f}s"),
            ]:
                c.drawString(50,y,f"{label}: {val}")
                y -= 20
            y -= 20
        c.save()

        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download PDF Report", f,
                file_name=f"MMBI_Engine_Report_{uploaded.name}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        os.unlink(video_path)
        st.success("✅ Analysis complete!")

# ============================================================
#  FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    🎯 MMBI Engine — AI-Powered Audience Engagement Analytics<br>
    Built with TensorFlow, OpenCV & Streamlit &nbsp;|&nbsp; Model: MobileNetV2, 65.0% test accuracy (FER-2013)
</div>
""", unsafe_allow_html=True)
