# 🔍 Age & Gender Detector — OpenCV DNN

Real-time face detection with **age estimation** and **gender classification** using
pre-trained deep learning models via OpenCV's DNN module. No PyTorch, no TensorFlow —
just OpenCV and NumPy.

---

## 📦 Requirements

```
Python 3.7+
opencv-python >= 4.5
numpy >= 1.21
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🛠 Setup (Download Models — one time only)

```bash
python setup_models.py
```

This downloads ~60 MB of Caffe model files into the `models/` folder:

| File | Purpose |
|------|---------|
| `face_deploy.prototxt` + `face.caffemodel` | OpenCV res10 SSD face detector |
| `age_deploy.prototxt` + `age_net.caffemodel` | Levi & Hassner age estimator |
| `gender_deploy.prototxt` + `gender_net.caffemodel` | Levi & Hassner gender classifier |

---

## 🚀 Usage

### Live webcam
```bash
python detector.py
```

### Static image
```bash
python detector.py --image photo.jpg
```
Saves result as `photo_detected.jpg`.

### Video file
```bash
python detector.py --video clip.mp4
```

### Different camera index
```bash
python detector.py --camera 1
```

### Adjust face confidence threshold (0–1)
```bash
python detector.py --conf 0.5    # lower = detect more faces (may include false positives)
python detector.py --conf 0.85   # higher = only very confident detections
```

---

## ⌨️ Controls (live/video modes)

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit |
| `S` | Save snapshot → `snapshot_NNN.jpg` |

---

## 🧠 How It Works

```
Frame
  │
  ▼
[Face Detector]  ← res10_300x300_ssd (Caffe)
  │  Bounding boxes
  ▼
[Crop + Pad face region]
  │
  ├──► [Age Net]    ← 8-bucket classifier  → (0-2) … (60-100)
  └──► [Gender Net] ← Binary classifier   → Male / Female
         │
         ▼
  Overlay labels + confidence bars on frame
```

### Models used

- **Face detection**: OpenCV's `res10_300x300_ssd_iter_140000_fp16` — fast, accurate SSD
  with a ResNet-10 backbone trained on mixed face datasets.

- **Age & Gender**: Levi & Hassner (2015) — lightweight CNNs trained on the Adience
  benchmark. Age output is one of **8 age-range buckets**, not a single number, which
  makes the model more robust to variation.

---

## 📁 Project Structure

```
age_gender_detector/
├── detector.py          ← main script
├── setup_models.py      ← model downloader
├── requirements.txt
├── README.md
└── models/              ← auto-created by setup_models.py
    ├── face_deploy.prototxt
    ├── face.caffemodel
    ├── age_deploy.prototxt
    ├── age_net.caffemodel
    ├── gender_deploy.prototxt
    └── gender_net.caffemodel
```

---

## ⚙️ Configuration (inside detector.py)

| Variable | Default | Description |
|----------|---------|-------------|
| `PADDING` | `20` | Extra pixels around face crop for better accuracy |
| `conf_threshold` | `0.7` | Min face detection confidence |
| `MODEL_MEAN` | Adience mean | BGR mean subtraction for age/gender nets |

---

## 📌 Notes & Tips

- **Better accuracy**: ensure good, even lighting and faces are relatively front-facing.
- **Multiple faces**: all detected faces in a frame are processed simultaneously.
- **Age buckets**: the model outputs a probability distribution across 8 age ranges;
  the highest-probability bucket is displayed.
- **GPU acceleration**: the script defaults to CPU. To use CUDA:
  ```python
  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
  net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
  ```
  Requires `opencv-contrib-python` built with CUDA support.

---

## 📚 References

- Levi, G. & Hassner, T. (2015). *Age and Gender Classification using Convolutional Neural Networks*. CVPR Workshop.
- OpenCV DNN Face Detector: https://github.com/opencv/opencv/tree/master/samples/dnn
