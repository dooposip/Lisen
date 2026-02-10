import cv2
import streamlit as st
import pandas as pd
from pathlib import Path


# ================================
# 3단계 위험 행동 라벨
# ================================
RISK_LABELS = {
    0: "안전",
    1: "경고",
    2: "위험"
}


def run_action_labeler(DATA_RAW, DATA_PROC):
    """🎬 행동 라벨링 (안전 / 경고 / 위험 + 영상 프레임 프리뷰 포함)"""

    # -----------------------------
    # 1) RAW 영상 스캔
    # -----------------------------
    videos = sorted(DATA_RAW.rglob("*.mp4"))
    if not videos:
        st.warning("📁 data/raw 폴더에 영상(mp4)이 없습니다.")
        return

    video_sel = st.selectbox("라벨링할 영상 선택", videos)

    # 영상 로드
    cap = cv2.VideoCapture(str(video_sel))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        st.error("영상을 열 수 없습니다.")
        return

    st.info(f"총 프레임 수: **{total_frames}**")

    # -----------------------------
    # 2) 프레임 프리뷰 표시
    # -----------------------------
    st.subheader("👀 프레임 미리보기")

    frame_idx = st.slider("프레임 선택", 0, total_frames - 1, 0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if ret:
        st.image(frame, channels="BGR", caption=f"Frame {frame_idx}", width=800)
    else:
        st.error("❌ 프레임 로드 실패")

    st.markdown("---")
    st.subheader("🎯 구간 라벨링 (안전 / 경고 / 위험)")

    # -----------------------------
    # 3) 저장 폴더 설정
    # -----------------------------
    save_dir = DATA_PROC / "action_labels"
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / f"{video_sel.stem}_action.csv"

    # 기존 라벨 CSV 있으면 불러오기
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["start_frame", "end_frame", "label"])

    # -----------------------------
    # 4) 라벨 구간 선택
    # -----------------------------
    col1, col2 = st.columns(2)
    with col1:
        start_f = st.number_input("시작 프레임", 0, total_frames - 1, frame_idx)
    with col2:
        end_f = st.number_input("종료 프레임", 0, total_frames - 1, frame_idx)

    label_name = st.radio(
        "위험도 선택",
        list(RISK_LABELS.values()),
        horizontal=True
    )
    label_id = [k for k, v in RISK_LABELS.items() if v == label_name][0]

    # -----------------------------
    # 5) 라벨 저장
    # -----------------------------
    if st.button("💾 구간 라벨 저장하기"):
        if start_f > end_f:
            st.error("❌ 시작 프레임은 종료 프레임보다 작아야 합니다.")
            return

        # 기존 구간 겹치는 부분 제거
        df = df[
            ~(
                (df["start_frame"] <= end_f) &
                (df["end_frame"] >= start_f)
            )
        ]

        df.loc[len(df)] = [start_f, end_f, label_id]
        df = df.sort_values("start_frame")
        df.to_csv(csv_path, index=False)

        st.success(f"📌 {start_f} ~ {end_f} 프레임 → '{label_name}' 저장 완료!")

    st.markdown("---")
    st.subheader("📄 저장된 라벨")

    if csv_path.exists():
        st.dataframe(df.reset_index(drop=True))
    else:
        st.info("아직 저장된 라벨이 없습니다.")