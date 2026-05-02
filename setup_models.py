"""
setup_models.py — Download all pre-trained model files
Run this once before using detector.py
"""

import os, sys, urllib.request

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = [
    # Face detection (OpenCV res10 SSD)
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
     "face_deploy.prototxt"),
    ("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel",
     "face.caffemodel"),
    # Age estimation (Levi & Hassner)
    ("https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/deploy_age.prototxt",
     "age_deploy.prototxt"),
    ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age_net.caffemodel",
     "age_net.caffemodel"),
    # Gender classification (Levi & Hassner)
    ("https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/deploy_gender.prototxt",
     "gender_deploy.prototxt"),
    ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender_net.caffemodel",
     "gender_net.caffemodel"),
]


def _progress(block, bsize, total):
    done = block * bsize
    if total > 0:
        pct = min(done * 100 / total, 100)
        bar = int(pct / 4)
        print(f"\r  [{'█'*bar}{'░'*(25-bar)}] {pct:5.1f}%  ({done//1024} KB)", end="", flush=True)


def main():
    print("=" * 55)
    print("  Model Setup for Age & Gender Detector")
    print("=" * 55)

    failed = []
    for url, filename in FILES:
        dest = os.path.join(MODEL_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print(f"  ✔  {filename:35s}  already downloaded")
            continue
        print(f"\n  ⬇  {filename}")
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress)
            size = os.path.getsize(dest) // 1024
            print(f"\n     ✔  {size} KB saved")
        except Exception as e:
            print(f"\n  ✖  FAILED: {e}")
            failed.append((filename, url))
            if os.path.exists(dest):
                os.remove(dest)

    print("\n" + "=" * 55)
    if not failed:
        print("  ✅  All models downloaded successfully!")
        print("  👉  Run: python detector.py")
    else:
        print(f"  ⚠   {len(failed)} file(s) failed to download:")
        for fname, url in failed:
            print(f"      {fname}")
            print(f"      URL: {url}")
        print("\n  Please download them manually and place in ./models/")
        sys.exit(1)
    print("=" * 55)


if __name__ == "__main__":
    main()
