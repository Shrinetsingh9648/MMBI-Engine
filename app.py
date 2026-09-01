import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
import tempfile
import os
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
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
    (col1, "🎭", "Emotion Detection", "MobileNetV2, 75.8% test accuracy on FER+"),
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
    # MediaPipe Face Detection — tighter, more accurate crops than Haar
    # cascades, which often clip faces off-center or include background.
    face_det = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )
    return model, class_names, face_det


def _fig_to_png_bytes(fig, dpi=150):
    """Saves a matplotlib figure to PNG bytes in memory (for embedding in the PDF)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()


def _draw_pdf_image(c, png_bytes, x, y_top, max_width, max_height):
    """
    Draws a PNG (from bytes) into a ReportLab canvas, scaled to fit within
    max_width x max_height while preserving aspect ratio. Returns the actual
    height drawn, so the caller can advance their y-cursor correctly.
    """
    img = ImageReader(BytesIO(png_bytes))
    iw, ih = img.getSize()
    scale = min(max_width / iw, max_height / ih)
    draw_w, draw_h = iw * scale, ih * scale
    c.drawImage(img, x, y_top - draw_h, width=draw_w, height=draw_h,
                preserveAspectRatio=True, mask='auto')
    return draw_h


def _ensure_pdf_space(c, y, needed, page_w, page_h, margin=40):
    """If the remaining space on the page is too small for the next block,
    starts a new page and returns the reset y-cursor."""
    if y - needed < margin:
        c.showPage()
        return page_h - 60
    return y


def _color_hist(frame_bgr, box):
    """
    Appearance descriptor for person re-identification. Uses Hue+Saturation
    from HSV (not raw BGR or grayscale) because H/S are far more stable
    under lighting changes than brightness — a person's hair/skin/clothing
    hue stays roughly the same whether they're brightly or dimly lit, unlike
    a plain grayscale intensity histogram, which was the old approach and
    is easily confused between people who happen to have similar brightness.
    """
    x, y, w, h = box
    roi = frame_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    # L1 (sum-to-1) normalization -- NOT NORM_MINMAX. MinMax is peak-relative,
    # which is extremely brittle for the sparse, single-peaked histograms
    # typical of a skin-tone-dominated face crop: a tiny shift in which exact
    # bin holds the peak (from ordinary compression/lighting noise) swings
    # the whole normalized histogram, making HISTCMP_CORREL nearly meaningless.
    # Verified directly against real footage: MINMAX gave a same-face,
    # one-frame-apart correlation of just 0.06 (should be ~1.0), which is
    # what caused duplicate identities to be created within the first few
    # frames of a video. L1 gives the correct ~1.0 for that same case.
    # alpha=1 is the TARGET SUM for NORM_L1 (beta is ignored) -- NOT the same
    # meaning as NORM_MINMAX's alpha=min/beta=max. Verified directly: with
    # alpha=0 (copied from the old MINMAX call without adjusting), every
    # histogram came out entirely zero, making EVERY comparison -- same
    # person or different person -- degenerate to a meaningless constant.
    cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
    return hist


def _match_frame_faces(faces, frame_bgr, people, frame_idx, cut_detected,
                        stale_frames, appearance_gate, fallback_threshold,
                        max_people, recycle_after_frames, iou_min=0.25,
                        absolute_max_people=None):
    """
    Decides which known person (if any) each detected face belongs to, for
    ONE frame — shared by both the initial scan and the main analysis loop
    so their matching behavior can't silently drift apart.

    Uses a GLOBALLY SORTED GREEDY assignment: every plausible (face, person)
    pairing in the frame is scored, then assignments are made strongest-
    match-first. This matters when two people are near each other — matching
    each face independently in whatever order they were detected can let an
    earlier face "steal" a person that was actually the better match for a
    different face, silently swapping two people's identities. Sorting by
    score first prevents that.

    SLOT RECYCLING: `max_people` limits how many people can be tracked
    SIMULTANEOUSLY, not the total across the whole video. If all slots are
    full and a new, unmatched face shows up, the ACTIVE person who's been
    absent the LONGEST is retired (stops being a match target) to free a
    slot — but only if their absence clearly exceeds a normal occlusion gap
    (`recycle_after_frames`, deliberately much longer than `stale_frames`),
    so a person who just looked away or was briefly covered keeps their
    reserved slot rather than losing it to someone else.

    The freed slot's id is REUSED for whoever takes it over, rather than
    minting a new one. This keeps the total number of distinct "Person N"
    identities ever reported capped at exactly `max_people`, matching what
    the "Max people to track" setting promises, and means tracking can
    always resume after a genuine long absence instead of permanently
    stopping once `max_people` identities have ever existed. The retired
    person's timeline/photo data (kept elsewhere, keyed by this same id) is
    left as-is when this happens; the caller uses the returned
    `is_new_person` flag as the signal to start a fresh timeline for that
    id, since it may now represent a different individual.

    Mutates `people` in place. Returns a list of (box, pid, is_new_person)
    for every face that got assigned; faces with nowhere to go (no match,
    no free slot even after checking for a recyclable one) are omitted.
    """
    if absolute_max_people is None:
        # This must equal max_people, not a multiple of it. It caps the TOTAL
        # number of distinct person identities ever created for the whole
        # video, not just how many can be active at once. Previously this was
        # max(8, max_people * 4), which let the tracker silently mint new
        # identities (via slot recycling after someone leaves/reappears) far
        # beyond what the "Max people to track" sidebar setting promised --
        # e.g. a slider set to 1 could still end up reporting 3+ people.
        absolute_max_people = max_people

    def _active_people():
        return {pid: p for pid, p in people.items() if p.get("active", True)}

    face_hists = []
    for box in faces:
        face_hists.append(_color_hist(frame_bgr, box))

    active = _active_people()
    candidates = []  # (score, face_idx, pid) -- higher score = better match
    for face_idx, box in enumerate(faces):
        hist = face_hists[face_idx]
        if hist is None:
            continue

        # Spatial candidates: gated by appearance so position alone is never
        # sufficient (this is what stopped different people merging when
        # they land in a similar screen spot) -- EXCEPT when IoU is so high
        # it's very unlikely to be anything but genuine continuity (see
        # HIGH_IOU_THRESHOLD below). Given a +1.0 offset so any valid
        # spatial match always outranks a pure-appearance one.
        #
        # TIERED GATE, calibrated against real measured footage: a video's
        # own brief fade-in/flash can make TRUE same-person appearance
        # correlation between adjacent frames drop as low as ~0.06 (verified
        # directly), while two visually similar people at a genuine cut can
        # still correlate as high as ~0.75 (also verified directly) -- these
        # ranges overlap, so appearance alone cannot cleanly separate them.
        # IoU does separate them in tested footage (0.965 for genuine
        # continuity vs 0.615 for a real cut), so very high spatial overlap
        # is trusted with almost no appearance confirmation needed.
        #
        # A THIRD, MODERATE tier was added after real footage showed the
        # original two-tier version losing tracking of a person who never
        # left frame or got cut away from: as someone turns their head or
        # lighting shifts slightly, IoU against their last box commonly
        # drops into the 0.4-0.85 range (still clearly continuous motion,
        # no cut) while appearance correlation dips as low as ~0.72 at the
        # same time -- both well under the 0.85 appearance_gate used for
        # HIGH_IOU misses, which caused the match to be rejected, the
        # person to go stale, and (since re-acquisition also requires
        # appearance_gate) tracking to permanently stop for the rest of
        # the video. Requiring the FULL appearance bar at moderate-but-real
        # spatial overlap was too strict for actual footage; a lower gate
        # here still protects against the two-people-swap case (which
        # shows much lower IoU, well under MODERATE_IOU_THRESHOLD) while
        # letting ordinary head movement keep matching.
        HIGH_IOU_THRESHOLD = 0.85
        HIGH_IOU_GATE = 0.0  # near pass-through -- position alone is the signal at this level of overlap
        MODERATE_IOU_THRESHOLD = 0.4
        MODERATE_IOU_GATE = 0.55
        if not cut_detected:
            for pid, pdata in active.items():
                if frame_idx - pdata["last_seen"] > stale_frames:
                    continue
                iou = _iou(box, pdata["last_box"])
                if iou > iou_min:
                    sim = cv2.compareHist(pdata["hist"], hist, cv2.HISTCMP_CORREL)
                    if iou > HIGH_IOU_THRESHOLD:
                        required_gate = HIGH_IOU_GATE
                    elif iou > MODERATE_IOU_THRESHOLD:
                        required_gate = MODERATE_IOU_GATE
                    else:
                        required_gate = appearance_gate
                    if sim > required_gate:
                        candidates.append((1.0 + iou + sim, face_idx, pid))

        # Appearance-only candidates: the fallback for first sightings,
        # re-appearances after a gap, or right after a detected cut.
        for pid, pdata in active.items():
            sim = cv2.compareHist(pdata["hist"], hist, cv2.HISTCMP_CORREL)
            if sim > fallback_threshold:
                candidates.append((sim, face_idx, pid))

    candidates.sort(key=lambda c: c[0], reverse=True)

    face_to_pid = {}
    used_pids = set()
    for score, face_idx, pid in candidates:
        if face_idx in face_to_pid or pid in used_pids:
            continue
        face_to_pid[face_idx] = pid
        used_pids.add(pid)

    results = []
    for face_idx, box in enumerate(faces):
        hist = face_hists[face_idx]
        if hist is None:
            continue
        x, y, w, h = box
        face_size = w * h
        is_new = False

        if face_idx in face_to_pid:
            pid = face_to_pid[face_idx]
        else:
            active = _active_people()  # recompute — may have changed via recycling below in this same frame
            if len(active) < max_people and len(people) < absolute_max_people:
                pid = len(people) + 1
                people[pid] = {"face": box, "frame": frame_bgr.copy(), "size": 0, "active": True,
                                "hist": hist, "last_box": box, "last_seen": frame_idx, "quality": 0}
                is_new = True
            else:
                # All slots full — check whether the longest-absent active
                # person has genuinely left (not just occluded) and can be
                # safely retired to make room, without touching their data.
                oldest_pid, oldest_gap = None, -1
                for cand_pid, cand_pdata in active.items():
                    gap = frame_idx - cand_pdata["last_seen"]
                    if gap > oldest_gap:
                        oldest_gap, oldest_pid = gap, cand_pid
                if oldest_pid is not None and oldest_gap > recycle_after_frames:
                    # Reuse the retired person's own id/slot rather than
                    # minting a new one. This keeps the total number of
                    # distinct "Person N" identities capped at exactly
                    # max_people (matching the "Max people to track"
                    # setting), and — critically — means tracking can
                    # always resume after a genuine long absence instead
                    # of permanently stopping once max_people identities
                    # have ever been created. Their prior timeline/photo
                    # data (kept elsewhere, keyed by this same pid) is
                    # intentionally left as-is; the caller treats is_new
                    # as the signal to start a fresh timeline for the pid.
                    people[oldest_pid] = {"face": box, "frame": frame_bgr.copy(), "size": 0, "active": True,
                                            "hist": hist, "last_box": box, "last_seen": frame_idx, "quality": 0}
                    pid = oldest_pid
                    is_new = True
                else:
                    continue  # everyone plausibly still around, no room for a new person this frame

        pdata = people[pid]
        pdata["last_box"] = box
        pdata["last_seen"] = frame_idx
        # The matching reference ("hist") is refreshed on EVERY match, not
        # just quality improvements. It previously only updated inside the
        # quality-gated block below, which ties it to "best representative
        # photo" bookkeeping -- meant for picking a clear snapshot to show
        # in the report, not for tracking continuity. That let it freeze on
        # whatever frame had the best size/sharpness early on and never
        # update again. As the person's appearance naturally drifted from
        # that frozen snapshot (lighting, head angle) over the following
        # minutes, similarity eventually dropped below the match threshold
        # for good, with nothing left to refresh it -- silently ending
        # tracking for the rest of the video even though the same person
        # was still clearly, continuously on screen. Updating on every
        # match instead tracks gradual drift the way a real re-id
        # comparison should: against the most recent known appearance.
        pdata["hist"] = hist
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if (roi := frame_bgr[y:y+h, x:x+w]).size > 0 else None
        quality = face_size * (1 + _sharpness(gray_roi))
        if quality > pdata.get("quality", 0):
            pdata.update(face=box, frame=frame_bgr.copy(), size=face_size, quality=quality)

        results.append((box, pid, is_new))

    return results


def _sharpness(gray_roi):
    """
    Measures image sharpness via Laplacian variance — a standard, cheap blur
    detector (blurry images have far less high-frequency edge detail, so the
    variance of the Laplacian drops sharply). Used to avoid picking a
    motion-blurred frame as someone's representative reference photo.
    """
    if gray_roi is None or gray_roi.size == 0:
        return 0.0
    return cv2.Laplacian(gray_roi, cv2.CV_64F).var()


def _iou(box_a, box_b):
    """Intersection-over-Union of two (x,y,w,h) boxes — used to detect duplicate face boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    intersection = iw * ih
    union = aw*ah + bw*bh - intersection
    return intersection / union if union > 0 else 0.0


def is_scene_cut(prev_gray_small, curr_gray_small, threshold=0.5):
    """
    Detects a hard cut (edited video switching camera angle/speaker) using a
    coarse whole-frame histogram comparison. During continuous footage
    (someone talking, moving their head), the background and overall frame
    stay similar frame-to-frame, so correlation stays high. A hard cut
    changes the whole scene at once, causing a sharp drop — this is a much
    stronger signal than trying to detect it from face position alone.
    """
    if prev_gray_small is None:
        return False
    h1 = cv2.calcHist([prev_gray_small], [0], None, [32], [0,256])
    h2 = cv2.calcHist([curr_gray_small], [0], None, [32], [0,256])
    cv2.normalize(h1, h1); cv2.normalize(h2, h2)
    corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return corr < threshold


def detect_faces(face_det, frame_bgr, dedup_iou_thresh=0.4):
    """
    Runs MediaPipe face detection on a BGR frame and returns a list of
    (x, y, w, h) pixel boxes — same shape the rest of the app already
    expects, so the multi-person tracking logic below didn't need to change.

    Also deduplicates near-identical overlapping boxes for the SAME real
    face (a known MediaPipe quirk) — without this, duplicate boxes were
    getting registered/tracked as two different people, causing both the
    "one person shows as two" and "two people get identical stats" bugs.
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_det.process(rgb)
    boxes = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            bw = min(int(bbox.width * w), w - x)
            bh = min(int(bbox.height * h), h - y)
            if bw > 0 and bh > 0:
                boxes.append((x, y, bw, bh))

    # Deduplicate: if two boxes overlap heavily, keep only the larger one
    boxes.sort(key=lambda b: b[2]*b[3], reverse=True)  # largest first
    deduped = []
    for box in boxes:
        if all(_iou(box, kept) < dedup_iou_thresh for kept in deduped):
            deduped.append(box)
    return deduped


INTEREST_MAP = {
    "happy": 1.0, "surprise": 1.0, "neutral": 0.75,
    "fear": 0.5, "sad": 0.25, "angry": 0.1, "disgust": 0.1,
}
PERSON_COLORS_BGR = [(0,200,255), (255,100,0), (0,255,100), (255,0,200)]
PERSON_COLORS_PLT = ["#00c8ff", "#ff6400", "#00ff64", "#ff00c8"]


def predict_emotion(model, class_names, face_roi_bgr):
    # Guard against degenerate crops (zero width/height near frame edges) —
    # these would otherwise reach the model with a broken shape and crash it.
    if face_roi_bgr is None or face_roi_bgr.size == 0 or 0 in face_roi_bgr.shape[:2]:
        return None, 0.0, None

    roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, IMG_SIZE)
    roi_in = preprocess_input(roi_resized.astype("float32"))
    roi_in = np.expand_dims(roi_in, axis=0)

    # One-time debug print of the shape actually being sent in vs. what the
    # model expects — check "Manage app" > Logs on Streamlit Cloud if a
    # ValueError still occurs, this line will show the real mismatch.
    if not st.session_state.get("_shape_logged", False):
        print(f"[MMBI DEBUG] model.input_shape={model.input_shape}  roi_in.shape={roi_in.shape}")
        st.session_state["_shape_logged"] = True

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
    """
    Keras 3 treats a nested sub-model (like our MobileNetV2 backbone) as an
    opaque callable — pulling a tensor out of its internal graph (the old
    approach) produces a graph that isn't connected to the outer model's
    inputs, and raises "Output ... is not connected to inputs".

    Fix: rebuild the forward pass explicitly on fresh Input tensors —
    1) a standalone model for the backbone up to its last conv layer
    2) manually replay the remaining "head" layers (GAP, Dropout, Dense, etc.)
    This produces a properly connected graph regardless of nesting.
    """
    backbone, conv_name = find_backbone_and_last_conv(_model)
    if backbone is None:
        return None

    try:
        # Standalone backbone-up-to-last-conv model (has its own clean graph)
        grad_backbone = tf.keras.Model(
            inputs=backbone.input,
            outputs=backbone.get_layer(conv_name).output,
        )

        inputs = tf.keras.Input(shape=_model.input_shape[1:])
        conv_output = grad_backbone(inputs)

        # Replay every layer that comes AFTER the backbone in the outer model
        x = conv_output
        found_backbone = False
        for layer in _model.layers:
            if layer is backbone:
                found_backbone = True
                continue
            if not found_backbone:
                continue  # skip the Input layer and anything before/including backbone
            x = layer(x)

        grad_model = tf.keras.Model(inputs, [conv_output, x])
        return grad_model
    except Exception as e:
        print(f"[MMBI DEBUG] build_gradcam_model failed: {e}")
        return None


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
    # matplotlib removed cm.get_cmap() in newer versions — matplotlib.colormaps[name] replaces it
    try:
        colormap = matplotlib.colormaps["jet"]
    except AttributeError:
        colormap = cm.get_cmap("jet")  # fallback for older matplotlib
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
    threshold  = st.slider("Detection sensitivity", 0.70, 0.98, 0.85,
                            help="How similar two face crops must look to be treated as the same person. Higher = stricter (less likely to merge two different people, but may split one person into two if lighting changes a lot). Recalibrated against real measured same/different-person similarity scores.")
    show_gradcam = st.checkbox("🔥 Show Grad-CAM explanations", value=True,
                                help="Highlights which parts of each face the AI focused on for its top prediction")
    generate_pdf = st.checkbox("📄 Generate PDF report", value=False,
                                help="Off by default — PDF generation adds extra processing time. Turn on if you need a downloadable report.")

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
    **Test accuracy: 75.8%** on FER+ (7-class)
    - Happy: 87.6% precision
    - Neutral: 81.9% precision
    - Surprise: 75.9% precision
    - Fear / Sad / Disgust: weakest classes (dataset-wide known issue — low sample counts and inherent visual overlap between these expressions)
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
        SPATIAL_STALE_FRAMES_SCAN = int(fps * 0.5)   # ~0.5s -- fps-aware so a 60fps video isn't 4x twitchier than a 15fps one
        RECYCLE_AFTER_FRAMES_SCAN = int(fps * 3)     # ~3s -- only free a slot for someone new after a much longer absence than the staleness window above, so a brief look-away during the scan doesn't cost someone their slot
        # Recalibrated against real measured data: with the histogram bug fixed,
        # same-person correlation sustains ~0.95-1.0, while different-person
        # correlation (even with a shared studio backdrop) tops out ~0.84-0.89.
        # 0.15 was calibrated against the OLD broken metric and let almost
        # anything through, which is what caused two different people to
        # merge into one tracked identity.
        # Recalibrated against real measured data: a genuine cut between two
        # visually-similar people measured 0.749 correlation in tested
        # footage, so this must sit above that. Genuine same-person
        # continuity (once past any fade-in) sustains 0.95-1.0, so 0.85
        # leaves comfortable margin on both sides. The fade-in edge case
        # (same person, but low correlation right at a video's start) is
        # handled separately by the tiered IoU rule above, not by this gate.
        APPEARANCE_SANITY_GATE = 0.85
        prev_gray_small = None

        for i in range(int(fps * scan_secs)):
            ret, frame = cap.read()
            if not ret: break
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (64, 64))
            cut_detected = is_scene_cut(prev_gray_small, gray_small)
            prev_gray_small = gray_small

            faces = detect_faces(face_det, frame)
            _match_frame_faces(faces, frame, people, i, cut_detected,
                                SPATIAL_STALE_FRAMES_SCAN, APPEARANCE_SANITY_GATE, threshold,
                                max_people, RECYCLE_AFTER_FRAMES_SCAN)
        cap.release()

        # ── Analyze video ─────────────────────────────────
        status.text("🎬 Analyzing video frame by frame...")
        timelines = {pid: [] for pid in people.keys()}
        best_frame_per_person = {pid: {"conf": 0.0} for pid in people.keys()}
        emotion_counts = {pid: {} for pid in people.keys()}
        cap, fcount = cv2.VideoCapture(video_path), 0
        SPATIAL_STALE_FRAMES_MAIN = 45  # ~3s of gap at frame-skip=2 before spatial proximity alone is no longer trusted
        RECYCLE_AFTER_FRAMES_MAIN = int(fps * 10)  # ~10s of absence -- long enough that this is "they left", not "they looked away"

        # Any person found during the scan phase should be treated as "just seen"
        # at the start of this loop, otherwise the staleness check below would
        # immediately consider them stale before analysis even begins.
        for pdata in people.values():
            pdata["last_seen"] = 0

        prev_gray_small = None

        while True:
            ret, frame = cap.read()
            if not ret: break
            ts   = fcount / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (64, 64))
            cut_detected = is_scene_cut(prev_gray_small, gray_small)
            prev_gray_small = gray_small

            if fcount % 2 == 0:
                faces = detect_faces(face_det, frame)
                matches = _match_frame_faces(faces, frame, people, fcount, cut_detected,
                                              SPATIAL_STALE_FRAMES_MAIN, APPEARANCE_SANITY_GATE, threshold,
                                              max_people, RECYCLE_AFTER_FRAMES_MAIN)
                for (x,y,w,h), best_pid, is_new in matches:
                    if is_new:
                        timelines[best_pid] = []
                        best_frame_per_person[best_pid] = {"conf": 0.0}
                        emotion_counts[best_pid] = {}

                    face_roi_bgr = frame[y:y+h, x:x+w]
                    try:
                        emotion, conf, face_rgb_96 = predict_emotion(model, class_names, face_roi_bgr)
                    except Exception as e:
                        # Don't let one bad frame crash the whole analysis —
                        # log it and just skip this detection.
                        print(f"[MMBI DEBUG] predict_emotion failed: {e}")
                        continue
                    if emotion is None:
                        continue  # degenerate crop, skip
                    if conf < 0.45: emotion = "neutral"
                    ew    = INTEREST_MAP.get(emotion, 0.5)
                    score = round(ew*100, 1)
                    timelines[best_pid].append({"time": round(ts,2), "score": score, "emotion": emotion})
                    emotion_counts[best_pid][emotion] = emotion_counts[best_pid].get(emotion, 0) + 1

                    # Blend confidence with sharpness so a confident-but-blurry
                    # frame (motion blur can still classify fine) doesn't win
                    # over a clearer one — sharpness is capped so it nudges the
                    # choice rather than overriding confidence entirely.
                    sharpness_val = _sharpness(cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2GRAY))
                    new_quality = conf * (1 + min(sharpness_val, 500) / 500)
                    if new_quality > best_frame_per_person[best_pid].get("quality", 0):
                        best_frame_per_person[best_pid] = {
                            "conf": conf, "emotion": emotion,
                            "face_rgb_96": face_rgb_96,
                            "pred_index": class_names.index(emotion) if emotion in class_names else 0,
                            "quality": new_quality,
                        }

            pct = int(fcount/total*100)
            progress.progress(min(pct,100))
            fcount += 1

        cap.release()
        status.text("✅ Analysis complete!")
        progress.progress(100)

        st.markdown(f"### 👥 Found {len(people)} people")
        if people:
            cols = st.columns(len(people))
            for pid, pdata in people.items():
                x,y,w,h = pdata["face"]
                fimg = cv2.cvtColor(pdata["frame"][y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                cols[pid-1].image(fimg, caption=f"Person {pid}", use_container_width=True)

        # ── Show results ───────────────────────────────────
        st.markdown("---")
        st.markdown("## 📊 Results")

        # Precompute each person's smoothed dataframe once, reused below
        person_dfs = {}
        for pid, tl in timelines.items():
            if not tl: continue
            df = pd.DataFrame(tl)
            df["sm"] = df["score"].rolling(30, min_periods=1).mean()
            person_dfs[pid] = df

        # ── Combined chart: everyone's timeline on one graph ──
        combined_chart_png = None
        if len(person_dfs) > 0:
            st.markdown("### 🆚 All People — Combined Timeline")
            figc, axc = plt.subplots(figsize=(12,4.5))
            figc.patch.set_facecolor("#1a1a2e")
            axc.set_facecolor("#16213e")
            axc.fill_between([0, max(df["time"].max() for df in person_dfs.values())], 65,100,alpha=0.10,color="green")
            axc.fill_between([0, max(df["time"].max() for df in person_dfs.values())], 40,65,alpha=0.10,color="yellow")
            axc.fill_between([0, max(df["time"].max() for df in person_dfs.values())], 0,40,alpha=0.10,color="red")
            for pid, df in person_dfs.items():
                axc.plot(df["time"], df["sm"], color=PERSON_COLORS_PLT[pid-1], lw=2.5, label=f"Person {pid}")
            axc.axhline(65, color="green", lw=0.8, ls="--")
            axc.axhline(40, color="orange", lw=0.8, ls="--")
            axc.set_xlim(0, max(df["time"].max() for df in person_dfs.values()))
            axc.set_ylim(0,100)
            axc.set_xlabel("Time(s)", color="white")
            axc.set_ylabel("Interest Score", color="white")
            axc.tick_params(colors="white")
            legend = axc.legend(facecolor="#1a1a2e", labelcolor="white")
            for sp in axc.spines.values(): sp.set_edgecolor("#555")
            st.pyplot(figc)
            if generate_pdf:
                combined_chart_png = _fig_to_png_bytes(figc)
            plt.close()
            st.markdown("---")

        pdf_data = {}
        for pid, tl in timelines.items():
            if not tl: continue
            st.markdown(f"### Person {pid}")
            df = person_dfs[pid]
            scores = df["score"].values
            avg   = np.mean(scores)
            inter = np.mean(scores>=65)*100
            neut  = np.mean((scores>=40)&(scores<65))*100
            not_i = np.mean(scores<40)*100
            pdf_data[pid] = dict(avg=avg, inter=inter, neut=neut, not_i=not_i,
                                  peak=max(scores), peak_t=tl[int(np.argmax(scores))]["time"])

            if generate_pdf:
                # Reference face image, for putting a face to the report
                px, py, pw, ph = people[pid]["face"]
                ref_crop_bgr = people[pid]["frame"][py:py+ph, px:px+pw]
                ok, buf = cv2.imencode(".png", ref_crop_bgr)
                if ok:
                    pdf_data[pid]["ref_face_png"] = buf.tobytes()

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Avg Score", f"{avg:.1f}/100")
            c2.metric("Interested", f"{inter:.1f}%")
            c3.metric("Neutral", f"{neut:.1f}%")
            c4.metric("Not Interested", f"{not_i:.1f}%")

            if avg >= 65:
                st.markdown('<div class="verdict-interested">✅ HIGHLY INTERESTED</div>', unsafe_allow_html=True)
                pdf_data[pid]["verdict"] = "HIGHLY INTERESTED"
            elif avg >= 40:
                st.markdown('<div class="verdict-neutral">➡️ NEUTRAL</div>', unsafe_allow_html=True)
                pdf_data[pid]["verdict"] = "NEUTRAL"
            else:
                st.markdown('<div class="verdict-not">❌ NOT INTERESTED</div>', unsafe_allow_html=True)
                pdf_data[pid]["verdict"] = "NOT INTERESTED"

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
            if generate_pdf:
                pdf_data[pid]["timeline_png"] = _fig_to_png_bytes(fig)
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
                    if generate_pdf:
                        pdf_data[pid]["emotion_png"] = _fig_to_png_bytes(fig2)
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
                        if generate_pdf:
                            ok, buf = cv2.imencode(".png", cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
                            if ok:
                                pdf_data[pid]["gradcam_png"] = buf.tobytes()
                                pdf_data[pid]["gradcam_caption"] = cap_txt
                    except Exception as e:
                        st.info(f"Grad-CAM unavailable for this model architecture ({e})")
                else:
                    if not show_gradcam:
                        st.info("Enable 'Show Grad-CAM explanations' in the sidebar to see this.")
                    elif grad_model is None:
                        st.info("Grad-CAM couldn't be built for this model — see app logs for details.")
                    else:
                        st.info("No confident frame available yet for this person.")

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

        # ── PDF report — optional, only runs if the sidebar checkbox is on ──
        if generate_pdf:
            st.markdown("---")
            try:
                import textwrap

                def _draw_checkbox(c, x, y, size=9):
                    c.rect(x, y, size, size, fill=0, stroke=1)

                pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
                c = canvas.Canvas(pdf_path, pagesize=A4)
                W, H = A4
                margin = 40
                content_w = W - 2*margin

                # ── Header ──
                c.setFillColorRGB(0.05,0.05,0.3)
                c.rect(0,H-90,W,90,fill=1,stroke=0)
                c.setFont("Helvetica-Bold",22)
                c.setFillColor(colors.white)
                c.drawString(margin,H-45,"MMBI Engine Report")
                c.setFont("Helvetica",11)
                c.drawString(margin,H-65, f"Video: {uploaded.name}  |  Date: {datetime.datetime.now().strftime('%d %b %Y %H:%M')}")
                c.setFont("Helvetica-Oblique", 9)
                c.drawString(margin, H-80, "AI-generated analysis -- please review and share feedback in each section below")

                y = H - 110

                # ── About / disclaimer, so this is understandable out of context ──
                c.setFont("Helvetica", 9)
                c.setFillColor(colors.grey)
                about = ("This report was generated automatically using a MobileNetV2 facial-emotion model "
                         "(75.8% test accuracy on the FER+ benchmark). It is an estimate of viewer interest "
                         "based on facial expressions, not a certainty -- please use the feedback sections "
                         "below to confirm whether it actually matches how the viewer felt.")
                for line in textwrap.wrap(about, 100):
                    y = _ensure_pdf_space(c, y, 14, W, H, margin)
                    c.drawString(margin, y, line)
                    y -= 12
                y -= 10

                # ── Combined chart (only meaningful with 2+ people) ──
                if combined_chart_png is not None and len(pdf_data) > 1:
                    y = _ensure_pdf_space(c, y, 200, W, H, margin)
                    c.setFont("Helvetica-Bold", 13)
                    c.setFillColor(colors.black)
                    c.drawString(margin, y, "All People -- Combined Timeline")
                    y -= 18
                    drawn_h = _draw_pdf_image(c, combined_chart_png, margin, y, content_w, 220)
                    y -= drawn_h + 20

                # ── Per-person sections ──
                verdict_color_map = {
                    "HIGHLY INTERESTED": colors.HexColor("#0a8a0a"),
                    "NEUTRAL": colors.HexColor("#b8860b"),
                    "NOT INTERESTED": colors.HexColor("#c0392b"),
                }

                for pid, d in pdf_data.items():
                    y = _ensure_pdf_space(c, y, 40, W, H, margin)
                    c.setFillColorRGB(0.1,0.1,0.5)
                    c.rect(margin, y-25, content_w, 30, fill=1, stroke=0)
                    c.setFont("Helvetica-Bold", 14)
                    c.setFillColor(colors.white)
                    c.drawString(margin+10, y-17, f"Person {pid}")
                    y -= 45

                    # Reference face + stats side-by-side
                    face_w = 100
                    y = _ensure_pdf_space(c, y, face_w + 10, W, H, margin)
                    row_top = y
                    if d.get("ref_face_png"):
                        _draw_pdf_image(c, d["ref_face_png"], margin, row_top, face_w, face_w)
                        stats_x = margin + face_w + 20
                    else:
                        stats_x = margin

                    stat_y = row_top - 14
                    c.setFont("Helvetica-Bold", 13)
                    c.setFillColor(verdict_color_map.get(d.get("verdict"), colors.black))
                    c.drawString(stats_x, stat_y, d.get("verdict", ""))
                    stat_y -= 20
                    c.setFont("Helvetica", 10)
                    c.setFillColor(colors.black)
                    for label, val in [
                        ("Average Score", f"{d['avg']:.1f}/100"),
                        ("Interested", f"{d['inter']:.1f}%"),
                        ("Neutral", f"{d['neut']:.1f}%"),
                        ("Not Interested", f"{d['not_i']:.1f}%"),
                        ("Peak Interest", f"{d['peak']:.0f}% at {d['peak_t']:.1f}s"),
                    ]:
                        c.drawString(stats_x, stat_y, f"{label}: {val}")
                        stat_y -= 15

                    y = row_top - face_w - 15

                    # Timeline chart
                    if d.get("timeline_png"):
                        y = _ensure_pdf_space(c, y, 180, W, H, margin)
                        c.setFont("Helvetica-Bold", 11)
                        c.setFillColor(colors.black)
                        c.drawString(margin, y, "Interest Timeline")
                        y -= 14
                        drawn_h = _draw_pdf_image(c, d["timeline_png"], margin, y, content_w, 170)
                        y -= drawn_h + 15

                    # Emotion distribution + Grad-CAM side by side
                    half_w = (content_w - 15) / 2
                    has_emo = bool(d.get("emotion_png"))
                    has_grad = bool(d.get("gradcam_png"))
                    if has_emo or has_grad:
                        y = _ensure_pdf_space(c, y, 170, W, H, margin)
                        row_top2 = y
                        c.setFont("Helvetica-Bold", 10)
                        c.setFillColor(colors.black)
                        drawn_h1 = drawn_h2 = 0
                        if has_emo:
                            c.drawString(margin, row_top2, "Emotion Distribution")
                            drawn_h1 = _draw_pdf_image(c, d["emotion_png"], margin, row_top2 - 12, half_w, 140)
                        if has_grad:
                            gx = margin + half_w + 15
                            c.drawString(gx, row_top2, "Grad-CAM Focus")
                            drawn_h2 = _draw_pdf_image(c, d["gradcam_png"], gx, row_top2 - 12, half_w, 140)
                        y = row_top2 - 12 - max(drawn_h1, drawn_h2) - 15

                    # Viewer feedback section -- the whole point of sending this to the person
                    fb_h = 100
                    y = _ensure_pdf_space(c, y, fb_h + 10, W, H, margin)
                    c.setStrokeColor(colors.HexColor("#888888"))
                    c.setLineWidth(1)
                    c.rect(margin, y-fb_h, content_w, fb_h, fill=0, stroke=1)
                    c.setFont("Helvetica-Bold", 11)
                    c.setFillColor(colors.black)
                    c.drawString(margin+10, y-16, "Viewer Feedback -- does this match how you actually felt?")
                    c.setFont("Helvetica", 10)
                    cb_y = y - 38
                    _draw_checkbox(c, margin+10, cb_y-7, 9)
                    c.drawString(margin+24, cb_y-9, "Yes, accurate")
                    _draw_checkbox(c, margin+150, cb_y-7, 9)
                    c.drawString(margin+164, cb_y-9, "Partially accurate")
                    _draw_checkbox(c, margin+330, cb_y-7, 9)
                    c.drawString(margin+344, cb_y-9, "Not accurate")
                    c.drawString(margin+10, y-60, "Comments:")
                    c.line(margin+80, y-61, margin+content_w-10, y-61)
                    c.line(margin+10, y-80, margin+content_w-10, y-80)
                    y -= fb_h + 25

                # Footer
                c.setFont("Helvetica", 8)
                c.setFillColor(colors.grey)
                c.drawString(margin, 25, "Generated by MMBI Engine -- AI-Powered Audience Engagement Analytics")
                c.save()

                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📄 Download PDF Report", f,
                        file_name=f"MMBI_Engine_Report_{uploaded.name}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            except Exception as e:
                # Surface the real error instead of failing silently or crashing the app
                st.error(f"PDF generation failed: {e}")
                print(f"[MMBI DEBUG] PDF generation failed: {e}")
        else:
            st.info("📄 PDF report generation is off. Enable it in the sidebar if you want a downloadable report.")

        os.unlink(video_path)
        st.success("✅ Analysis complete!")

# ============================================================
#  FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    🎯 MMBI Engine — AI-Powered Audience Engagement Analytics<br>
    Built with TensorFlow, OpenCV & Streamlit &nbsp;|&nbsp; Model: MobileNetV2, 75.8% test accuracy (FER+)
</div>
""", unsafe_allow_html=True)
