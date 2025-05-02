import os
import csv
import random
import warnings

# Suppress TensorFlow INFO/WARN logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings(
    "ignore",
    message=".*torch.cuda.amp.autocast.*",
    category=FutureWarning
)

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

# Import your pipeline components
from test1 import (
    model,
    device,
    val_tfm,
    emotion_labels,         # e.g. ["neutral","happy","sad",…]
    analyze_scene_with_clip,
    detect_objects,
    refiner,
    detector
)

DATASET_ROOT = "dataset"
IMAGE_ROOT   = os.path.join(DATASET_ROOT, "val")
LABEL_CSV    = "labels.csv"

# 1) Build labels.csv
subfolders = sorted(
    d for d in os.listdir(IMAGE_ROOT)
    if os.path.isdir(os.path.join(IMAGE_ROOT, d))
)
with open(LABEL_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename","emotion"])
    for emo in subfolders:
        emo_dir = os.path.join(IMAGE_ROOT, emo)
        for fn in sorted(os.listdir(emo_dir)):
            if fn.lower().endswith((".jpg",".jpeg",".png")):
                writer.writerow([os.path.join(emo, fn), emo])
print(f"Wrote {LABEL_CSV} with classes: {subfolders}")

label_map = {
    "Angry":   "anger",
    "Disgust": "disgust",
    "Fear":    "fear",
    "Happy":   "happy",
    "Neutral": "neutral",
    "Sad":     "sad",
    "Surprise":"surprise"
}

def evaluate_and_visualize():
    # Load ground-truth
    gt_map = {}
    with open(LABEL_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for fn, lbl in reader:
            gt_map[fn] = label_map.get(lbl, lbl.lower())
    print(f"Found {len(gt_map)} labeled images\n")

    all_gts, all_prs, init_preds = [], [], []
    results = []
    refiner.base_confidence = 60.0

    for fn, true_lbl in tqdm(gt_map.items(), desc="Evaluating", unit="img"):
        path = os.path.join(IMAGE_ROOT, fn)
        frame = cv2.imread(path)
        if frame is None:
            continue

        # Scene + objects
        scene = analyze_scene_with_clip(frame)
        det   = detect_objects(frame)
        objs  = det[0] if isinstance(det, tuple) else det

        # Face detection
        try:
            faces = detector.detect_faces(frame)
        except:
            faces = []

        # Initial prediction
        if not faces:
            init, conf = "neutral", 0.0
        else:
            x,y,w,h = faces[0]['box']
            x,y = max(0,x), max(0,y)
            w,h = min(w,frame.shape[1]-x), min(h,frame.shape[0]-y)
            crop = frame[y:y+h, x:x+w]
            pil  = Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).convert("L")
            tensor = val_tfm(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                logits    = model(tensor)
                probs     = F.softmax(logits, dim=1)
                conf_val, idx = probs.max(1)
                init      = emotion_labels[idx.item()]
                conf      = conf_val.item() * 100

        init_preds.append(init)

        # Refined prediction
        refined = refiner.refine_emotion_with_clip(init, objs, conf, scene)
        all_gts.append(true_lbl)
        all_prs.append(refined)

        results.append({
            "filename": fn,
            "gt":       true_lbl,
            "initial":  init,
            "refined":  refined
        })

    # 4) Distributions
    print("Raw (initial) distribution:")
    for emo, c in Counter(init_preds).most_common():
        print(f"  {emo:>8s}: {c}")
    print("\nRefined distribution:")
    for emo, c in Counter(all_prs).most_common():
        print(f"  {emo:>8s}: {c}")
    print()

    # 5) Classification reports
    print("Classification Report (Initial Predictions):")
    print(classification_report(all_gts, init_preds, labels=emotion_labels))
    print("\nClassification Report (Post-Refinement):")
    print(classification_report(all_gts, all_prs, labels=emotion_labels))

    # 6a) Confusion matrix for raw predictions
    cm_raw = confusion_matrix(all_gts, init_preds, labels=emotion_labels)
    print("\nConfusion Matrix (Initial Predictions):")
    print(cm_raw)   # <--- print array
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_raw, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(emotion_labels)))
    ax.set_yticks(np.arange(len(emotion_labels)))
    ax.set_xticklabels(emotion_labels, rotation=45, ha="right")
    ax.set_yticklabels(emotion_labels)
    ax.set_xlabel("Predicted (Initial)")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Initial Predictions)")
    for i in range(len(emotion_labels)):
        for j in range(len(emotion_labels)):
            ax.text(j, i, cm_raw[i, j],
                    ha="center", va="center",
                    color="white" if cm_raw[i, j] > cm_raw.max()/2 else "black")
    plt.tight_layout()
    plt.show(block=True)

    # 6b) Confusion matrix for refined predictions
    cm_refined = confusion_matrix(all_gts, all_prs, labels=emotion_labels)
    print("\nConfusion Matrix (Post-Refinement):")
    print(cm_refined)   # <--- print array
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_refined, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(emotion_labels)))
    ax.set_yticks(np.arange(len(emotion_labels)))
    ax.set_xticklabels(emotion_labels, rotation=45, ha="right")
    ax.set_yticklabels(emotion_labels)
    ax.set_xlabel("Predicted (Refined)")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Post-Refinement)")
    for i in range(len(emotion_labels)):
        for j in range(len(emotion_labels)):
            ax.text(j, i, cm_refined[i, j],
                    ha="center", va="center",
                    color="white" if cm_refined[i, j] > cm_refined.max()/2 else "black")
    plt.tight_layout()
    plt.show(block=True)

    # 7) Example grid of refined results
    examples = random.sample(results, min(4, len(results)))
    fig2, axes = plt.subplots(1, len(examples), figsize=(4*len(examples), 4))
    for ax, ex in zip(axes, examples):
        img = cv2.imread(os.path.join(IMAGE_ROOT, ex["filename"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"GT: {ex['gt']}\nPred: {ex['refined']}", fontsize=10)
    plt.tight_layout()
    plt.show(block=True)

if __name__ == "__main__":
    evaluate_and_visualize()
