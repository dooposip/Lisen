import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import cv2


# ---------------------------------------------------------
# YOLO pose label → keypoints (x,y)
# ---------------------------------------------------------
def parse_pose_label(label_path: Path):
    if not label_path.exists():
        return None

    with open(label_path, "r") as f:
        line = f.readline().strip()

    if not line:
        return None

    parts = line.split()

    if len(parts) < 56:
        return None

    kps = parts[5:]  # x y v 반복

    arr = []
    for i in range(0, len(kps), 3):
        try:
            x = float(kps[i])
            y = float(kps[i + 1])
        except Exception:
            return None
        arr.append([x, y])

    return np.array(arr, dtype=float).reshape(-1)   # (34,)


# ---------------------------------------------------------
# pose txt 파일 찾기
# ---------------------------------------------------------
def find_pose_label_file(label_dirs, video_stem: str, frame_idx: int) -> Path | None:
    base = f"{video_stem}_{frame_idx:06d}"

    best = None
    for ld in label_dirs:
        original = ld / f"{base}.txt"
        if original.exists():
            return original

        cands = sorted(ld.glob(f"{base}_*.txt"))
        if cands and best is None:
            best = cands[0]

    return best


# ---------------------------------------------------------
# ACTION LABEL → frame 단위로 확장
# start_frame ~ end_frame 구간에 label 지정
# ---------------------------------------------------------
def expand_action_labels(df, total_frames):
    frame_labels = np.zeros(total_frames, dtype=int)  # 기본=0(안전)

    for _, row in df.iterrows():
        s = int(row["start_frame"])
        e = int(row["end_frame"])
        label = int(row["label"])

        s = max(s, 0)
        e = min(e, total_frames - 1)

        frame_labels[s:e + 1] = label

    return frame_labels


# ---------------------------------------------------------
# 행동 시퀀스 생성 (3단계 라벨 대응)
# ---------------------------------------------------------
def build_action_sequences(DATA_PROC: Path, FEATURE_DIR: Path | None = None, seq_len: int = 12):

    st.header("🧬 행동 학습용 Pose 시퀀스 생성 (3단계 전용)")

    # 1) action CSV 위치
    action_dir = DATA_PROC / "action_labels"
    if not action_dir.exists():
        st.warning("⚠ action_labels 폴더 없음")
        return

    csv_files = sorted(action_dir.glob("*.csv"))
    if not csv_files:
        st.warning("⚠ CSV 없음")
        return

    # 2) Pose 라벨 위치
    label_train = DATA_PROC / "labels" / "train"
    label_val = DATA_PROC / "labels" / "val"
    label_dirs = [p for p in [label_train, label_val] if p.exists()]
    if not label_dirs:
        st.error("❌ labels/train, labels/val 없음")
        return

    X_list, Y_list = [], []
    missing = 0

    # CSV마다 처리
    for csv in csv_files:
        df = pd.read_csv(csv)

        # CSV 형식 체크
        if not {"start_frame", "end_frame", "label"}.issubset(df.columns):
            st.error(f"❌ CSV 형식 오류: {csv.name}")
            return

        # video 이름 추출
        video_stem = csv.stem.replace("_action", "")

        # 총 프레임 수 추정
        max_frame = int(df["end_frame"].max()) + 1

        # frame 단위 위험도 라벨 확장
        frame_labels = expand_action_labels(df, max_frame)

        # 시퀀스 생성
        for start in range(0, max_frame - seq_len):
            seq_frames = list(range(start, start + seq_len))

            kps_seq = []
            valid = True

            for fr in seq_frames:
                pose_file = find_pose_label_file(label_dirs, video_stem, fr)
                if pose_file is None:
                    missing += 1
                    valid = False
                    break

                kps = parse_pose_label(pose_file)
                if kps is None:
                    missing += 1
                    valid = False
                    break

                kps_seq.append(kps)

            if not valid:
                continue

            X_list.append(np.array(kps_seq))
            Y_list.append(frame_labels[start + seq_len - 1])

    if not X_list:
        st.error("❌ 시퀀스 생성 실패: pose 또는 라벨 매칭 실패")
        return

    X = np.array(X_list, dtype=float)
    Y = np.array(Y_list, dtype=int)

    save_dir = DATA_PROC / "action_sequences"
    save_dir.mkdir(exist_ok=True)

    out = save_dir / "dataset_action_pose.npz"
    np.savez(out, X=X, Y=Y)

    st.success(f"🎉 시퀀스 {len(X)}개 생성 완료!")
    st.info(f"📁 저장 위치: {out}")