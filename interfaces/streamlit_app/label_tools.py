import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


# =========================================================
# Adult / Child 분류 (최종 안정 버전)
#   - 박스 면적, 높이, keypoint conf 모두 고려
#   - 애매한 프레임은 Adult 없음 → Child only (스킵 처리)
# =========================================================
def classify_adult_child(boxes_xyxy, kpts_conf=None, min_ratio=1.15):
    """
    boxes_xyxy : (N, 4)
    kpts_conf  : (N, K) or None
    min_ratio  : Adult 후보가 확실히 더 커야 하는 비율
    """
    if len(boxes_xyxy) == 0:
        return []

    areas = []
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = b
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)

        area = w * h
        score = area

        # keypoint 평균 신뢰도 반영
        if kpts_conf is not None:
            conf = float(np.mean(kpts_conf[i]))
            score *= max(conf, 0.4)

        areas.append((i, score, h))

    # 크기(score) 기준으로 정렬
    areas_sorted = sorted(areas, key=lambda x: x[1], reverse=True)

    # Adult 후보 1, 2의 스코어 비교
    if len(areas_sorted) >= 2:
        s1 = areas_sorted[0][1]
        s2 = areas_sorted[1][1]
        ratio = s1 / (s2 + 1e-6)

        # 차이가 애매 -> Adult 없음
        if ratio < min_ratio:
            return ["Child" for _ in boxes_xyxy]

    adult_idx = areas_sorted[0][0]

    labels = []
    for i in range(len(boxes_xyxy)):
        labels.append("Adult" if i == adult_idx else "Child")

    return labels


# =========================================================
# IOU 계산
# =========================================================
def compute_iou(a, b):
    x1, y1, x2, y2 = a
    X1, Y1, X2, Y2 = b
    inter_x1 = max(x1, X1)
    inter_y1 = max(y1, Y1)
    inter_x2 = min(x2, X2)
    inter_y2 = min(y2, Y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, x2 - x1) * max(0, y2 - y1)
    area_b = max(0, X2 - X1) * max(0, Y2 - Y1)
    return inter_area / (area_a + area_b - inter_area + 1e-6)


# =========================================================
# Keypoints clamp (bbox 안으로 제한)
# =========================================================
def clamp_keypoints_to_bbox(kpts, box):
    x1, y1, x2, y2 = box
    kpts[:, 0] = np.clip(kpts[:, 0], x1, x2)
    kpts[:, 1] = np.clip(kpts[:, 1], y1, y2)
    return kpts


# =========================================================
# 중복 및 품질 필터링
# =========================================================
def clean_detections(boxes_xyxy, boxes_xywhn, kpts_pixel, kpts_conf, stats):
    keep = list(range(len(boxes_xyxy)))

    # IOU 기반 중복 제거
    for i in range(len(boxes_xyxy)):
        for j in range(i + 1, len(boxes_xyxy)):
            if j not in keep:
                continue

            iou = compute_iou(boxes_xyxy[i], boxes_xyxy[j])
            if iou > 0.7:
                h_i = boxes_xyxy[i][3] - boxes_xyxy[i][1]
                h_j = boxes_xyxy[j][3] - boxes_xyxy[j][1]
                drop = j if h_i >= h_j else i
                if drop in keep:
                    keep.remove(drop)
                    stats["dropped_iou"] += 1

    final = []
    for bi in keep:
        # keypoint confidence 필터
        if kpts_conf is not None:
            conf = kpts_conf[bi]
            if np.mean(conf) < 0.5 or np.sum(conf > 0.5) < 8:
                stats["dropped_conf"] += 1
                continue

        # 좌우/상하 움직임 적은 탐지 제거
        kp_std = np.std(kpts_pixel[bi], axis=0)
        if kp_std[0] < 2 and kp_std[1] < 2:
            stats["dropped_kp"] += 1
            continue

        final.append(bi)

    return final


# =========================================================
# Adult/Child 텍스트 표시
# =========================================================
def put_class_text(img, box, name):
    x1, y1, x2, y2 = box.astype(int)
    color = (255, 0, 0) if name == "Adult" else (0, 255, 0)
    cv2.putText(img, name, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# =========================================================
# 어깨 중앙 root 보정
# =========================================================
def inject_mid_shoulder_root(kpts):
    if len(kpts) < 7:
        return kpts

    left = kpts[5]
    right = kpts[6]
    mid_x = (left[0] + right[0]) / 2
    mid_y = (left[1] + right[1]) / 2

    kpts[0][0] = mid_x
    kpts[0][1] = mid_y
    return kpts


# =========================================================
# 품질 기록
# =========================================================
def record_quality_stats(analysis_dir, stats):
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "label_quality.csv"
    df_new = pd.DataFrame([stats])
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(csv_path, index=False)


# =========================================================
# 라벨 생성 스트림 (YOLO Pose → Adult/Child + Keypoints 저장)
# =========================================================
def generate_yolo_pose_labels_stream(video_path, stride, model_path, save_root):

    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(video_path))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    img_train = save_root / "images/train"
    img_val = save_root / "images/val"
    lbl_train = save_root / "labels/train"
    lbl_val = save_root / "labels/val"
    analysis_dir = save_root / "analysis"

    for p in [img_train, img_val, lbl_train, lbl_val, analysis_dir]:
        p.mkdir(parents=True, exist_ok=True)

    stats = {
        "video": Path(video_path).name,
        "total_frames": 0,
        "frames_used_for_labeling": 0,
        "raw_dets": 0,
        "saved_dets": 0,
        "dropped_iou": 0,
        "dropped_conf": 0,
        "dropped_kp": 0,
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ================================================
        # 1) 실제 프레임 번호 (핵심 수정)
        # ================================================
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # stride 적용
        if frame_idx % stride != 0:
            continue

        stats["total_frames"] += 1

        pred = model(frame, verbose=False)[0]
        annotated = pred.plot()

        boxes_xyxy = pred.boxes.xyxy.cpu().numpy()
        boxes_xywhn = pred.boxes.xywhn.cpu().numpy()
        kpts_pixel = pred.keypoints.xy.cpu().numpy()
        kpts_conf = pred.keypoints.conf.cpu().numpy() if pred.keypoints.conf is not None else None

        stats["raw_dets"] += len(boxes_xyxy)

        # Adult/Child 분류
        raw_labels = classify_adult_child(boxes_xyxy, kpts_conf)

        # Adult가 1명만 확실할 때만 기록
        if raw_labels.count("Adult") != 1:
            continue

        # 품질 필터링
        final_indices = clean_detections(boxes_xyxy, boxes_xywhn, kpts_pixel, kpts_conf, stats)

        for bi in final_indices:
            put_class_text(annotated, boxes_xyxy[bi], raw_labels[bi])

        for bi in final_indices:
            kpts_pixel[bi] = inject_mid_shoulder_root(kpts_pixel[bi])
            kpts_pixel[bi] = clamp_keypoints_to_bbox(kpts_pixel[bi], boxes_xyxy[bi])

        # ================================================
        # 2) 프레임 번호 기반 stem 생성 (핵심 수정)
        # ================================================
        stem = f"{Path(video_path).stem}_{frame_idx:06d}.jpg"

        # train/val split
        if np.random.rand() < 0.8:
            img_path = img_train / stem
            lbl_path = lbl_train / (stem.replace(".jpg", ".txt"))
        else:
            img_path = img_val / stem
            lbl_path = lbl_val / (stem.replace(".jpg", ".txt"))

        # 이미지 저장
        if not img_path.exists():
            cv2.imwrite(str(img_path), frame)

        # ================================================
        # 3) txt 저장 (append, 핵심 수정)
        # ================================================
        stats["frames_used_for_labeling"] += 1
        h, w = frame.shape[:2]

        with open(lbl_path, "a") as f:  # ← append 모드
            for bi in final_indices:
                stats["saved_dets"] += 1
                cls = 0 if raw_labels[bi] == "Adult" else 1
                x, y, bw, bh = boxes_xywhn[bi]
                f.write(f"{cls} {x} {y} {bw} {bh}")
                for kp in kpts_pixel[bi]:
                    nx = kp[0] / w
                    ny = kp[1] / h
                    f.write(f" {nx:.6f} {ny:.6f} 2")
                f.write("\n")

        yield annotated, frame_idx, total

    cap.release()
    record_quality_stats(analysis_dir, stats)

# =========================================================
# 라벨 시각화
# =========================================================
def visualize_label(img_path, label_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, "이미지 읽기 실패"

    h, w = img.shape[:2]
    if not Path(label_path).exists():
        return img, "라벨 없음"

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls = int(parts[0])
        x, y, bw, bh = map(float, parts[1:5])
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        color = (255, 0, 0) if cls == 0 else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name = "Adult" if cls == 0 else "Child"
        cv2.putText(img, name, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if len(parts) > 5:
            kp = parts[5:]
            for i in range(0, len(kp), 3):
                nx = float(kp[i])
                ny = float(kp[i + 1])
                px = int(nx * w)
                py = int(ny * h)
                cv2.circle(img, (px, py), 4, (0, 255, 255), -1)

    return img, "라벨 시각화 완료"