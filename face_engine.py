"""
Face engine: YOLOv8 (face weights) for detection + ArcFace (InsightFace) for recognition.
Everything is open source and runs on CPU.
"""
import os
import pickle
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
from ultralytics import YOLO

DATA_DIR = Path("data")
KNOWN_DIR = DATA_DIR / "known_faces"      # data/known_faces/<person_name>/*.jpg
DB_PATH = DATA_DIR / "embeddings.pkl"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Passive anti-spoofing (real vs. photo/screen), 128x128 RGB input, logits [real, spoof]
LIVENESS_PATH = Path("models") / "liveness.onnx"
LIVENESS_URL = "https://github.com/facenox/face-antispoof-onnx/raw/main/models/best_model_quantized.onnx"
LIVENESS_SIZE = 128
LIVENESS_EXPAND = 1.5

DATA_DIR.mkdir(exist_ok=True)
KNOWN_DIR.mkdir(parents=True, exist_ok=True)


class FaceEngine:
    def __init__(self):
        # YOLOv8 face detector (downloaded once, then cached)
        weights = hf_hub_download("arnabdhar/YOLOv8-Face-Detection", "model.pt")
        self.detector = YOLO(weights)

        # ArcFace embeddings (512-d). Only detection+recognition modules are loaded.
        self.embedder = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.embedder.prepare(ctx_id=-1, det_size=(320, 320))

        if not LIVENESS_PATH.exists():
            LIVENESS_PATH.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(LIVENESS_URL, LIVENESS_PATH)
        self.liveness = ort.InferenceSession(str(LIVENESS_PATH), providers=["CPUExecutionProvider"])
        self._live_input = self.liveness.get_inputs()[0].name

        self.db = self._load_db()  # {name: [embedding, ...]}

    # ---------- database ----------
    def _load_db(self):
        if DB_PATH.exists():
            with open(DB_PATH, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_db(self):
        with open(DB_PATH, "wb") as f:
            pickle.dump(self.db, f)

    def people(self):
        return {n: len(v) for n, v in self.db.items()}

    def delete_person(self, name):
        self.db.pop(name, None)
        self._save_db()

    # ---------- detection / embedding ----------
    def detect(self, img_bgr, conf=0.5):
        res = self.detector.predict(img_bgr, conf=conf, verbose=False)[0]
        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            boxes.append((max(x1, 0), max(y1, 0), x2, y2, float(b.conf[0])))
        return boxes

    def embed_box(self, img_bgr, box):
        """Crop the YOLO box with some padding and get its ArcFace embedding."""
        x1, y1, x2, y2 = box[:4]
        h, w = img_bgr.shape[:2]
        pw, ph = int((x2 - x1) * 0.3), int((y2 - y1) * 0.3)
        crop = img_bgr[max(y1 - ph, 0):min(y2 + ph, h), max(x1 - pw, 0):min(x2 + pw, w)]
        if crop.size == 0:
            return None
        faces = self.embedder.get(crop)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding

    # ---------- liveness ----------
    def liveness_score(self, img_bgr, box):
        """Return real-minus-spoof logit for a face box (higher = more likely a live face)."""
        x1, y1, x2, y2 = box[:4]
        h, w = img_bgr.shape[:2]
        side = int(max(x2 - x1, y2 - y1) * LIVENESS_EXPAND)
        if side <= 0:
            return -99.0
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        sx, sy = cx - side // 2, cy - side // 2
        # reflect-pad so crops near the image border keep the face centered
        pad = side
        padded = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        crop = padded[sy + pad:sy + pad + side, sx + pad:sx + pad + side]
        if crop.size == 0:
            return -99.0
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (LIVENESS_SIZE, LIVENESS_SIZE), interpolation=cv2.INTER_AREA)
        inp = (rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)[None]
        real, spoof = self.liveness.run([], {self._live_input: inp})[0][0]
        return float(real - spoof)

    # ---------- enrollment ----------
    def enroll(self, name, images_bgr):
        """Add one or more images of a person. Returns number of faces successfully added."""
        added = 0
        for img in images_bgr:
            boxes = self.detect(img)
            if not boxes:
                continue
            best = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            emb = self.embed_box(img, best)
            if emb is not None:
                self.db.setdefault(name, []).append(emb)
                added += 1
        if added:
            self._save_db()
        return added

    def build_from_folder(self, folder=KNOWN_DIR):
        """Index saved images: one sub-folder per person."""
        self.db = {}
        report = {}
        for person_dir in sorted(Path(folder).iterdir()):
            if not person_dir.is_dir():
                continue
            imgs = []
            for p in person_dir.iterdir():
                if p.suffix.lower() in IMG_EXT:
                    im = cv2.imread(str(p))
                    if im is not None:
                        imgs.append(im)
            report[person_dir.name] = self.enroll(person_dir.name, imgs)
        self._save_db()
        return report

    # ---------- recognition ----------
    def identify(self, img_bgr, conf=0.5, threshold=0.4, live_threshold=0.0):
        """Detect all faces and match each against the database (cosine similarity).

        Faces failing the liveness check are reported as spoofs and never matched.
        """
        results = []
        for box in self.detect(img_bgr, conf):
            live = self.liveness_score(img_bgr, box)
            if live < live_threshold:
                results.append({"box": box[:4], "det_conf": box[4], "name": "Spoof",
                                "score": 0.0, "live": live})
                continue
            emb = self.embed_box(img_bgr, box)
            name, score = "Unknown", 0.0
            if emb is not None and self.db:
                for person, embs in self.db.items():
                    s = float(np.max(np.dot(np.array(embs), emb)))
                    if s > score:
                        name, score = person, s
                if score < threshold:
                    name = "Unknown"
            results.append({"box": box[:4], "det_conf": box[4], "name": name,
                            "score": score, "live": live})
        return results

    @staticmethod
    def annotate(img_bgr, results):
        out = img_bgr.copy()
        for r in results:
            x1, y1, x2, y2 = r["box"]
            spoof = r["name"] == "Spoof"
            known = r["name"] not in ("Unknown", "Spoof")
            color = (0, 200, 0) if known else (0, 140, 255) if spoof else (0, 0, 220)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f'{r["name"]} {r["score"]:.2f}' if known else "SPOOF" if spoof else "Unknown"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
        return out
