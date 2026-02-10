# interfaces/streamlit_app/data_validation.py

from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import cv2


# ===================================================
# 1) 이미지 / 라벨 매칭 검사
# ===================================================
def validate_image_label_pairs(DATA_PROC):
    img_train = DATA_PROC / "images/train"
    lbl_train = DATA_PROC / "labels/train"
    img_val = DATA_PROC / "images/val"
    lbl_val = DATA_PROC / "labels/val"

    def scan(img_dir, lbl_dir):
        imgs = {p.stem for p in img_dir.glob("*.jpg")}
        labels = {p.stem for p in lbl_dir.glob("*.txt")}
        return imgs - labels, labels - imgs

    return {
        "train": scan(img_train, lbl_train),
        "val": scan(img_val, lbl_val)
    }


# ===================================================
# 2) YOLO 라벨 값 검증
# ===================================================
def validate_label_values(label_path):
    errors = []
    with open(label_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()

        if len(parts) < 5:
            errors.append(f"Line {i}: Not enough values")
            continue

        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:5])

        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            errors.append(f"Line {i}: bbox out of range")

        # Keypoints 검사
        if len(parts) > 5:
            kp = parts[5:]
            if len(kp) % 3 != 0:
                errors.append(f"Line {i}: invalid keypoint count")

            for k in range(0, len(kp), 3):
                try:
                    nx = float(kp[k])
                    ny = float(kp[k + 1])
                except:
                    errors.append(f"Line {i}: keypoint parse failed")
                    break

                if not (0 <= nx <= 1 and 0 <= ny <= 1):
                    errors.append(f"Line {i}: keypoint out of range")
                    break

    return errors


# ===================================================
# 3) Bounding Box 클래스 분포 카운트
# ===================================================
def collect_class_distribution(DATA_PROC):
    lbl_train = DATA_PROC / "labels/train"
    lbl_val = DATA_PROC / "labels/val"

    counter = Counter()

    for lbl in list(lbl_train.glob("*.txt")) + list(lbl_val.glob("*.txt")):
        with open(lbl, "r") as f:
            for line in f.readlines():
                cls = int(line.split()[0])
                counter[cls] += 1

    return counter


# ===================================================
# 4) 행동(Action) 라벨 CSV 검증
# ===================================================
def validate_action_labels(DATA_RAW, DATA_PROC):

    action_dir = DATA_PROC / "action_labels"
    if not action_dir.exists():
        return {"error": "action_labels 폴더 없음"}

    csv_files = sorted(action_dir.glob("*.csv"))
    if not csv_files:
        return {"error": "CSV 파일 없음"}

    results = {}

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        video_name = csv_path.stem.replace("_action", "")
        video_path = list(DATA_RAW.rglob(video_name + ".mp4"))

        report = {
            "duplicate_frames": 0,
            "out_of_range_frames": 0,
            "invalid_action_id": 0,
            "not_sorted": False
        }

        # 영상 프레임 수 확인
        if video_path:
            cap = cv2.VideoCapture(str(video_path[0]))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            total_frames = None

        frames = df["frame"].tolist()
        actions = df["action_id"].tolist()

        # 정렬 검사
        if frames != sorted(frames):
            report["not_sorted"] = True

        # 중복 검사
        report["duplicate_frames"] = len(frames) - len(set(frames))

        # 범위 검사
        if total_frames is not None:
            report["out_of_range_frames"] = sum([1 for f in frames if f >= total_frames])

        # 행동 ID 검사
        valid_ids = {0, 1, 2, 3, 4, 5, 6}
        report["invalid_action_id"] = sum([1 for a in actions if a not in valid_ids])

        results[csv_path.name] = report

    return results