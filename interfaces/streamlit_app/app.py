# ============================================================================
# LiSEN Dashboard (Fixed Import + Fixed Path Version)
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from transformer import TransformerClassifier
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2] 

# 주요 경로들
DATA_PROC  = PROJECT_ROOT / "data" / "processed"
DATA_RAW   = PROJECT_ROOT / "data" / "raw"
MODEL_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
FEATURE_DIR  = DATA_PROC / "features"
best_model_path = PROJECT_ROOT / "models" / "best.pt"
transformer_model_path = PROJECT_ROOT / "models" / "transformer_action_pose.pt"
data_raw_path = PROJECT_ROOT / "data" / "raw"

# Python import 경로에 프로젝트 루트 강제 등록
sys.path.append(str(PROJECT_ROOT))

# 존재하지 않으면 자동 생성
for d in [DATA_PROC, DATA_RAW, MODEL_DIR, RESULTS_DIR, FEATURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="LiSEN Workflow Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# Custom Module Imports
# =========================================================
from interfaces.streamlit_app.Helper import (
    scan_raw_data,
    render_data_overview,
    render_project_summary,
)

from interfaces.streamlit_app.label_tools import (
    generate_yolo_pose_labels_stream,
    visualize_label
)

from interfaces.streamlit_app.training_tools import (
    yolo_pose_training_tab
)

from interfaces.streamlit_app.data_validation import (
    validate_image_label_pairs,
    validate_label_values,
    collect_class_distribution,
)

from dataset_augmentation import (
    load_sequence_dataset,
    apply_sequence_augmentations,
    save_augmented_dataset
)

from interfaces.streamlit_app.action_labeler import run_action_labeler
from interfaces.streamlit_app.action_dataset_builder import build_action_sequences
from interfaces.streamlit_app.transformer import train_transformer_model
from interfaces.streamlit_app.info import (
    show_model_inference_tab,
    show_frame_labeling_tab,
)

# =====================================================================
# sidebar 
# =====================================================================
from sidebar import render_sidebar
render_sidebar()

st.title("🧬 LiSEN")


# -------------------------------------------------------------
# 🔥 Welcome 화면 + 그라데이션 배경
# -------------------------------------------------------------
import streamlit as st

# 페이지 전체 배경에 그라데이션 적용
st.markdown("""
<style>
/* 메인 컨테이너 전체에 그라데이션 */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e3f2fd, #fce4ec);
    background-attachment: fixed;
}

/* 상단 헤더 영역(탭, 타이틀 자리)도 그라데이션 적용 */
[data-testid="stHeader"] {
    background: linear-gradient(135deg, #e3f2fd, #fce4ec) !important;
}

/* Streamlit 상단 decoration(흰색 얇은 선) 없애기 */
[data-testid="stDecoration"] {
    background: linear-gradient(135deg, #e3f2fd, #fce4ec) !important;
}

/* 헤더 아래 남는 공백 영역 제거 */
header[data-testid="stHeader"] {
    height: 60px;
}

/* 메인 컨테이너 배경 투명하게 유지 */
.block-container {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


if "show_tabs" not in st.session_state:
    st.session_state.show_tabs = False

if not st.session_state.show_tabs:

    st.markdown(
"""
<div style='text-align:center; padding-top:50px;'>

<h1 style='font-size:48px; font-weight:800;'>👋 환영합니다!</h1>

<h3 style='color:#555; margin-top:-10px; font-weight:600;'>
LiSEN AI Workflow Dashboard
</h3>

<p style='font-size:18px; color:#444; margin-top:25px;'>
YOLO Pose · Transformer 기반 위험도 분석 AI 워크플로우입니다.
</p>

<img src='https://em-content.zobj.net/source/microsoft-teams/337/robot_1f916.png'
     width='130' style='margin-top:30px; opacity:0.95;'>

<p style='font-size:17px; color:#555; margin-top:30px;'>
아래 버튼을 눌러 워크플로우를 시작하세요.
</p>

</div>
""",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([2,4,2])
    with col2:
        if st.button("🚀 시작하기", use_container_width=True):
            st.session_state.show_tabs = True
            st.rerun()

    st.stop()

# =========================================================
# Tabs
# =========================================================
TABS = st.tabs([
    "0️⃣ 튜토리얼",
    "1️⃣ Pose 라벨링",
    "2️⃣ 데이터 검증",
    "3️⃣ YOLO 학습",
    "4️⃣ 행동 라벨링",
    "5️⃣ 시퀀스 생성",
    "6️⃣ 데이터 증강",
    "7️⃣ Transformer 학습",
    "8️⃣ 모델 추론",
    "9️⃣ 실시간 위험 감지",
    "🔟 게시판",
])


# =====================================================================
# 0️⃣ 원본 데이터 탐색
# =====================================================================
with TABS[0]:
    st.header("📁 프로젝트 요약")
    render_project_summary()

    st.markdown("---")

    st.header("📊 데이터 탐색")
    stats = scan_raw_data(DATA_RAW)
    render_data_overview(stats)

# =====================================================================
# 1️⃣ Pose 라벨링
# =====================================================================
with TABS[1]:
    st.header("🏷 Pose 라벨링 (Adult/Child + Keypoints)")

    videos = sorted(DATA_RAW.rglob("*.mp4"))
    if not videos:
        st.info("📁 data/raw/ 아래에 mp4 파일을 넣어주세요.")
    else:
        video_sel = st.selectbox("라벨 생성할 영상", videos)
        stride = st.slider("Stride (프레임 간격)", 1, 15, 1)

        if "stop_label" not in st.session_state:
            st.session_state.stop_label = False

        start_btn, stop_btn = st.columns([3, 1])
        start = start_btn.button("▶ 라벨 생성")
        stop_btn.button("■ 중지", on_click=lambda: st.session_state.update(stop_label=True))

        if start:
            st.session_state.stop_label = False
            pose_model = MODEL_DIR / "yolo11m-pose.pt"

            if not pose_model.exists():
                st.error("❌ YOLO Pose 모델(yolo11m-pose.pt)이 없습니다.")
            else:
                stframe = st.empty()
                bar = st.progress(0)
                status = st.empty()

                for annotated, idx, total in generate_yolo_pose_labels_stream(
                    video_sel, stride, pose_model, DATA_PROC
                ):
                    if st.session_state.stop_label:
                        st.warning("🛑 라벨 생성 중지됨")
                        break

                    stframe.image(annotated, channels="BGR", width=450)
                    bar.progress(int((idx + 1) / total * 100))
                    status.text(f"{idx + 1}/{total}")

                else:
                    st.success("🎉 라벨 생성 완료!")

    st.markdown("---")
    st.subheader("👀 라벨 시각화")

    split = st.selectbox("Dataset", ["train", "val"])
    img_dir = DATA_PROC / f"images/{split}"
    lbl_dir = DATA_PROC / f"labels/{split}"

    images = sorted(img_dir.glob("*.jpg"))
    if images:
        img_sel = st.selectbox("이미지 선택", images)
        if st.button("시각화 보기"):
            vis, msg = visualize_label(img_sel, lbl_dir / f"{img_sel.stem}.txt")
            st.info(msg)
            st.image(vis, channels="BGR", width=450)


# =====================================================================
# 2️⃣ 데이터 검증
# =====================================================================
with TABS[2]:
    st.header("🧹 데이터 검증 / 정제")

    if st.button("🧪 검증 실행"):
        st.subheader("🔍 이미지 / 라벨 매칭")
        st.json(validate_image_label_pairs(DATA_PROC))

        st.markdown("---")
        st.subheader("🔎 YOLO 라벨 값 검증")

        errors = {}
        for split in ["train", "val"]:
            for lbl in (DATA_PROC / f"labels/{split}").glob("*.txt"):
                err = validate_label_values(lbl)
                if err:
                    errors[str(lbl)] = err

        if errors:
            st.error("⚠ 오류 라벨 발견")
            st.json(errors)
        else:
            st.success("✔ 모든 라벨 정상")

        st.markdown("---")
        st.subheader("📊 Adult/Child 클래스 분포")

        dist = collect_class_distribution(DATA_PROC)
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.bar(dist.keys(), dist.values(), color="#3498db")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Adult", "Child"])
        st.pyplot(fig)


# =====================================================================
# 3️⃣ YOLO 학습
# =====================================================================
with TABS[3]:
    st.header("🧠 Adult/Child 모델 학습")
    yolo_pose_training_tab(DATA_PROC, MODEL_DIR, RESULTS_DIR)


# =====================================================================
# 4️⃣ 행동 라벨링
# =====================================================================
with TABS[4]:
    st.header("🎬 행동(Action) 라벨링")
    run_action_labeler(DATA_RAW, DATA_PROC)


# =====================================================================
# 5️⃣ 시퀀스 생성
# =====================================================================
with TABS[5]:
    st.header("📈 시퀀스 생성 (Action Pose Sequence)")

    if st.button("▶ 시퀀스 생성 실행"):
        build_action_sequences(DATA_PROC, FEATURE_DIR)
        st.success("시퀀스 생성 완료!")


# =====================================================================
# 6️⃣ 데이터 증강
# =====================================================================
with TABS[6]:
    st.header("🧪 시퀀스 데이터 증강 ")

    seq_file = DATA_PROC / "action_sequences" / "dataset_action_pose.npz"

    if not seq_file.exists():
        st.error(f"❌ 시퀀스 파일 없음: {seq_file}")
        st.stop()

    X, Y = load_sequence_dataset(seq_file)

    st.write(f"**시퀀스 개수:** {len(X)}")
    st.write(f"**시퀀스 길이:** {X.shape[1]}")
    st.write(f"**Feature 수:** {X.shape[2]}")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:  use_flip = st.checkbox("좌우 반전", True)
    with col2:  use_jitter = st.checkbox("노이즈", True)
    with col3:  use_scale = st.checkbox("스케일", True)
    with col4:  use_shift = st.checkbox("Shift", True)

    aug_count = st.slider("증강 횟수", 1, 10, 3)

    if st.button("🚀 시퀀스 증강 실행"):
        with st.spinner("증강 생성 중..."):
            X_aug, Y_aug = apply_sequence_augmentations(
                X, Y,
                aug_count=aug_count,
                use_flip=use_flip,
                use_jitter=use_jitter,
                use_scale=use_scale,
                use_shift=use_shift
            )

            X_final = np.concatenate([X, X_aug], axis=0)
            Y_final = np.concatenate([Y, Y_aug], axis=0)

            save_path = DATA_PROC / "action_sequences" / "dataset_action_pose_aug.npz"
            save_augmented_dataset(X_final, Y_final, save_path)

        st.success(f"🎉 저장됨 → {save_path.name}")


# =====================================================================
# 7️⃣ Transformer 학습
# =====================================================================
with TABS[7]:
    st.header("🧠 Transformer 위험 행동 모델 학습")

    seq_file = DATA_PROC / "action_sequences" / "dataset_action_pose.npz"
    save_path = MODEL_DIR / "transformer_action_risk.pt"
    log_path = MODEL_DIR / "transformer_training_log.txt"

    if st.button("⚡ Transformer 학습 시작"):
        if not seq_file.exists():
            st.error("❌ 데이터셋 없음")
        else:
            with st.spinner("학습 중..."):
                acc = train_transformer_model(seq_file, save_path, log_file=log_path)
            st.success(f"🎉 완료! 정확도 = {acc:.3f}")

# =====================================================================
# 8️⃣ 모델 추론 
# =====================================================================
with TABS[8]:
    selected_video = show_model_inference_tab(MODEL_DIR, DATA_PROC)
    if selected_video:
        show_frame_labeling_tab(selected_video)

# =====================================================================
# 9️⃣ 실시간 위험 감지 (분석된 영상 기반)
# =====================================================================

with TABS[9]:

    st.header("🎬 9번탭: 분석된 영상 기반 위험도 모니터링")

    PREVIEW_DIR = RESULTS_DIR / "preview"
    
    if not PREVIEW_DIR.exists():
        st.warning("⚠️ 'results/preview' 디렉토리가 존재하지 않습니다.")
    else:
        preview_videos = sorted(list(PREVIEW_DIR.glob("preview_*.mp4")))
        
        if not preview_videos:
            st.info("ℹ️ 'results/preview' 디렉토리에 분석된 영상이 없습니다.")
        else:
            video_paths = [str(p) for p in preview_videos]
            selected_preview_video_path = st.selectbox(
                "분석된 영상 선택",
                video_paths
            )
            st.video(selected_preview_video_path)

# =====================================================================
# 🔟 게시판 (JSON 기반 저장 + 삭제 기능)
# =====================================================================

import json

with TABS[10]:
    st.header("📌 게시판")

    BOARD_PATH = PROJECT_ROOT / "board.json"

    # 게시판 파일 없으면 생성
    if not BOARD_PATH.exists():
        with open(BOARD_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=4)

    # 게시판 데이터 로딩
    with open(BOARD_PATH, "r", encoding="utf-8") as f:
        board_data = json.load(f)

    # ---- 글쓰기 ---
    st.subheader("✏ 게시글 작성")
    title = st.text_input("제목")
    content = st.text_area("내용")

    if st.button("📝 글 저장"):
        new_post = {
            "title": title,
            "content": content,
        }
        board_data.append(new_post)

        with open(BOARD_PATH, "w", encoding="utf-8") as f:
            json.dump(board_data, f, ensure_ascii=False, indent=4)

        st.success("📌 글이 저장되었습니다!")
        st.rerun()

    st.markdown("---")

    # ---- 글 목록 ---
    st.subheader("📚 게시글 목록")

    if len(board_data) == 0:
        st.info("아직 작성된 글이 없습니다.")
    else:
        # 게시글 제목 목록
        titles = [p["title"] for p in board_data]

        # 글 선택
        selected = st.selectbox("글 선택", options=titles)

        # 선택한 글 찾기
        for idx, post in enumerate(board_data):
            if post["title"] == selected:
                st.markdown(f"### 📝 {post['title']}")
                st.write(post["content"])

                # 삭제 버튼
                if st.button("🗑️ 이 글 삭제하기"):
                    board_data.pop(idx)

                    with open(BOARD_PATH, "w", encoding="utf-8") as f:
                        json.dump(board_data, f, ensure_ascii=False, indent=4)

                    st.success("🗑 글이 삭제되었습니다!")
                    st.rerun()
                break