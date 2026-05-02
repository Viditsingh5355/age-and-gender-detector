"""
Age & Gender Detector using OpenCV DNN
========================================
Uses pre-trained Caffe models:
  - Face detection:  res10_300x300_ssd (OpenCV)
  - Age estimation:  Levi & Hassner CNN
  - Gender classify: Levi & Hassner CNN

Run:
    python detector.py                 # live webcam
    python detector.py --image photo.jpg
    python detector.py --video clip.mp4
"""

import cv2
import numpy as np
import argparse
import sys
import os
import urllib.request
import time

# ─── Model URLs & paths ────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODELS = {
    "face_proto": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "path": os.path.join(MODEL_DIR, "face_deploy.prototxt"),
    },
    "face_model": {
        "url": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel",
        "path": os.path.join(MODEL_DIR, "face.caffemodel"),
    },
    "age_proto": {
        "url": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/deploy_age.prototxt",
        "path": os.path.join(MODEL_DIR, "age_deploy.prototxt"),
    },
    "age_model": {
        "url": "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age_net.caffemodel",
        "path": os.path.join(MODEL_DIR, "age_net.caffemodel"),
    },
    "gender_proto": {
        "url": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/deploy_gender.prototxt",
        "path": os.path.join(MODEL_DIR, "gender_deploy.prototxt"),
    },
    "gender_model": {
        "url": "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender_net.caffemodel",
        "path": os.path.join(MODEL_DIR, "gender_net.caffemodel"),
    },
}

# Age buckets used by the Levi & Hassner model
AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)"
]

GENDER_LIST = ["Male", "Female"]

# Visual theme
COLORS = {
    "Male":   (255, 170,  50),   # warm blue-ish gold
    "Female": ( 80, 200, 255),   # soft coral-pink
    "box":    (  0, 220, 120),   # bright mint
    "text_bg":(  0,   0,   0),
}

PADDING = 20   # pixels to expand face crop for better model accuracy


# ─── Download helpers ──────────────────────────────────────────────────────────

def _progress(block, block_size, total):
    downloaded = block * block_size
    if total > 0:
        pct = min(downloaded * 100 / total, 100)
        bar = int(pct / 5)
        print(f"\r  [{'█'*bar}{'░'*(20-bar)}] {pct:5.1f}%", end="", flush=True)


def download_models():
    """Download all required model files if not already present."""
    all_present = all(os.path.exists(m["path"]) and os.path.getsize(m["path"]) > 1000
                      for m in MODELS.values())
    if all_present:
        return

    print("\n📥  Downloading pre-trained models (first-time setup) …")
    for name, info in MODELS.items():
        path = info["path"]
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f"  ✔  {os.path.basename(path)} already present")
            continue
        print(f"  ⬇  {os.path.basename(path)}")
        try:
            urllib.request.urlretrieve(info["url"], path, reporthook=_progress)
            print()
        except Exception as e:
            print(f"\n  ✖  Failed: {e}")
            print(f"     Manual URL: {info['url']}")
            print(f"     Save to:    {path}")
    print()


# ─── Load networks ─────────────────────────────────────────────────────────────

def load_networks():
    download_models()

    face_net    = cv2.dnn.readNet(MODELS["face_proto"]["path"],  MODELS["face_model"]["path"])
    age_net     = cv2.dnn.readNet(MODELS["age_proto"]["path"],   MODELS["age_model"]["path"])
    gender_net  = cv2.dnn.readNet(MODELS["gender_proto"]["path"],MODELS["gender_model"]["path"])

    # Use GPU if available
    for net in (face_net, age_net, gender_net):
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("✅  Networks loaded successfully\n")
    return face_net, age_net, gender_net


# ─── Detection logic ───────────────────────────────────────────────────────────

MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)


def detect_faces(frame, face_net, conf_threshold=0.7):
    """Return list of (x1,y1,x2,y2) bounding boxes."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            boxes.append((max(0, x1), max(0, y1),
                          min(w - 1, x2), min(h - 1, y2)))
    return boxes


def predict_age_gender(face_img, age_net, gender_net):
    """Run age & gender inference on a cropped face image."""
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 MODEL_MEAN, swapRB=False)

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender_idx = gender_preds[0].argmax()
    gender = GENDER_LIST[gender_idx]
    gender_conf = float(gender_preds[0][gender_idx])

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_idx = age_preds[0].argmax()
    age = AGE_BUCKETS[age_idx]
    age_conf = float(age_preds[0][age_idx])

    return gender, gender_conf, age, age_conf


# ─── Drawing helpers ───────────────────────────────────────────────────────────

def draw_label(frame, text, pos, color, font_scale=0.65, thickness=2):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 4, y - th - 6), (x + tw + 4, y + baseline), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_confidence_bar(frame, x1, y1, conf, color, width=80, height=8):
    bar_x = x1
    bar_y = y1 - 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + width, bar_y + height), (50, 50, 50), -1)
    filled = int(conf * width)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + height), color, -1)


def annotate_frame(frame, boxes, age_net, gender_net):
    """Detect age/gender for each box and draw annotations."""
    h, w = frame.shape[:2]

    for (x1, y1, x2, y2) in boxes:
        # Expand crop with padding
        fx1 = max(0, x1 - PADDING)
        fy1 = max(0, y1 - PADDING)
        fx2 = min(w - 1, x2 + PADDING)
        fy2 = min(h - 1, y2 + PADDING)
        face_crop = frame[fy1:fy2, fx1:fx2]

        if face_crop.size == 0:
            continue

        gender, g_conf, age, a_conf = predict_age_gender(face_crop, age_net, gender_net)
        color = COLORS[gender]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        corner = 16
        thick  = 3
        for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame,(cx,cy),(cx+dx*corner,cy),color,thick)
            cv2.line(frame,(cx,cy),(cx,cy+dy*corner),color,thick)

        # Labels
        label = f"{gender} ({g_conf*100:.0f}%)  Age {age}"
        draw_label(frame, label, (x1, y1 - 10), color)

        # Confidence bar under label
        draw_confidence_bar(frame, x1, y1 - 28, g_conf, color)

    return frame


# ─── Header overlay ────────────────────────────────────────────────────────────

def draw_hud(frame, fps=None, face_count=0):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    title = "🔍 AGE & GENDER DETECTOR  |  OpenCV DNN"
    cv2.putText(frame, title, (10, 25), cv2.FONT_HERSHEY_DUPLEX,
                0.55, (200, 200, 200), 1, cv2.LINE_AA)

    if fps is not None:
        fps_text = f"FPS: {fps:5.1f}  |  Faces: {face_count}"
        cv2.putText(frame, fps_text, (w - 200, 25), cv2.FONT_HERSHEY_DUPLEX,
                    0.5, (100, 255, 160), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q / ESC = Quit  |  S = Save snapshot",
                (10, h - 10), cv2.FONT_HERSHEY_PLAIN,
                0.95, (120, 120, 120), 1, cv2.LINE_AA)


# ─── Processing modes ──────────────────────────────────────────────────────────

def process_image(path, face_net, age_net, gender_net):
    frame = cv2.imread(path)
    if frame is None:
        print(f"❌  Cannot open image: {path}")
        sys.exit(1)

    boxes = detect_faces(frame, face_net)
    annotate_frame(frame, boxes, age_net, gender_net)
    draw_hud(frame, face_count=len(boxes))

    out_path = os.path.splitext(path)[0] + "_detected.jpg"
    cv2.imwrite(out_path, frame)
    print(f"✅  Saved → {out_path}")

    cv2.imshow("Age & Gender Detector", frame)
    print("Press any key to close…")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_stream(source, face_net, age_net, gender_net):
    """Webcam (source=0) or video file."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌  Cannot open source: {source}")
        sys.exit(1)

    snap_count = 0
    prev_time  = time.time()
    fps        = 0.0

    print("▶  Stream running — press Q / ESC to quit, S to save snapshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_faces(frame, face_net)
        annotate_frame(frame, boxes, age_net, gender_net)

        now  = time.time()
        fps  = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        draw_hud(frame, fps=fps, face_count=len(boxes))
        cv2.imshow("Age & Gender Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q or ESC
            break
        if key in (ord('s'), ord('S')):
            snap_name = f"snapshot_{snap_count:03d}.jpg"
            cv2.imwrite(snap_name, frame)
            print(f"📸  Snapshot saved → {snap_name}")
            snap_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅  Done.")


# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time Age & Gender Detection with OpenCV DNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--image",  help="Path to an image file")
    parser.add_argument("--video",  help="Path to a video file")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--conf",   type=float, default=0.7,
                        help="Face detection confidence threshold (default: 0.7)")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════╗")
    print("║   Age & Gender Detector  —  OpenCV DNN  ║")
    print("╚══════════════════════════════════════════╝\n")

    face_net, age_net, gender_net = load_networks()

    if args.image:
        process_image(args.image, face_net, age_net, gender_net)
    elif args.video:
        process_stream(args.video, face_net, age_net, gender_net)
    else:
        process_stream(args.camera, face_net, age_net, gender_net)


if __name__ == "__main__":
    main()
