# =====================================================================
# 🧠 YOLO Pose Adult/Child Training (Corrected Version)
# =====================================================================

import streamlit as st
import torch
from ultralytics import YOLO
from pathlib import Path

def yolo_pose_training_tab(DATA_PROC: Path, MODEL_DIR: Path, RESULTS_DIR: Path):

    st.subheader("")

    # Dataset Paths
    train_dir = DATA_PROC / "images/train"
    val_dir = DATA_PROC / "images/val"
    yaml_path = DATA_PROC / "pose_dataset.yaml"
    labels_train = DATA_PROC / "labels/train"
    labels_val = DATA_PROC / "labels/val"

    # ============================
    # 1) Dataset Summary
    # ============================
    def get_imgs(path: Path):
        if not path.exists(): return []
        exts = (".jpg", ".png", ".jpeg")
        return [p for p in path.iterdir() if p.suffix.lower() in exts]

    train_imgs = get_imgs(train_dir)
    val_imgs = get_imgs(val_dir)

    c1, c2 = st.columns(2)
    c1.metric("Train 이미지", len(train_imgs))
    c2.metric("Val 이미지", len(val_imgs))

    # ============================
    # 2) dataset.yaml 생성
    # ============================
    yaml_text = f"""
train: {train_dir.as_posix()}
val: {val_dir.as_posix()}

nc: 2
names:
  0: Adult
  1: Child

kpt_shape: [17, 3]
"""
    yaml_path.write_text(yaml_text.strip(), encoding="utf-8")
    st.info(f"📄 pose_dataset.yaml 생성됨 → {yaml_path}")

    # ============================
    # 3) Base Pose Model 선택
    # ============================
    st.markdown("### 🔧 사전 학습 Pose 모델 선택")

    available_pose_models = [
        p for p in MODEL_DIR.glob("*.pt")
        if "pose" in p.name.lower()
    ]

    if not available_pose_models:
        st.error("❌ pose 모델(.pt)이 없습니다!\n`yolov8m-pose.pt` 또는 `yolo11m-pose.pt` 파일을 MODEL_DIR 안에 넣으세요.")
        return

    weights_path = st.selectbox(
        "📌 사용할 Pose Base Model",
        available_pose_models
    )

    st.success(f"사용 모델: {weights_path.name}")

    # ============================
    # 4) Hyperparameters
    # ============================
    epochs = st.number_input("Epochs", 1, 300, 100)
    imgsz = st.selectbox("Image Size", [320, 480, 640], index=2)
    batch = st.number_input("Batch Size", 1, 64, 16)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ============================
    # 5) Train Button
    # ============================
    if st.button("🚀 YOLO Pose 학습 시작"):

        try:
            st.info(f"⚙ Pose 모델 로딩 중: {weights_path}")
            model = YOLO(str(weights_path))

            st.info("🏋️‍♂️ 학습 시작...")

            model.train(
                data=str(yaml_path),
                epochs=int(epochs),
                imgsz=int(imgsz),
                batch=int(batch),
                device=device,
                project=str(RESULTS_DIR / "pose_train"),
                name="adult_child",
                exist_ok=True
            )

            st.success("🎉 YOLO Pose Adult/Child 학습 완료!")

        except Exception as e:
            st.error(f"❌ 학습 오류: {e}")