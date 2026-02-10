import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
from transformer import TransformerClassifier
import pandas as pd
import io
import tempfile

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =========================================================
# YOLO Pose → 34차원 추출
# =========================================================
def extract_pose_vector(result):
    try:
        if (not hasattr(result, "keypoints")) or (result.keypoints is None):
            return None

        kps_xy = result.keypoints.xyn[0].cpu().numpy()
        flat = kps_xy.flatten().astype(np.float32)

        return flat if len(flat) == 34 else None
    except:
        return None


# =========================================================
# Pose overlay
# =========================================================
def draw_pose_on_image(img, r_pose, names):

    pose_boxes = []

    if r_pose.boxes is None or r_pose.keypoints is None:
        return img, []

    for kp, box in zip(r_pose.keypoints, r_pose.boxes):
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{names[cls]} {conf:.2f}"

        if "Adult" in label:
            box_color = (0, 128, 255)
        else:
            box_color = (0, 255, 0)

        pose_boxes.append((x1, y1, x2, y2, label, box_color))

        kp_xy = kp.xy[0].cpu().numpy()
        for (x, y) in kp_xy:
            if x > 0 and y > 0:
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

    pose_boxes = sorted(pose_boxes, key=lambda x: x[1])
    used_y = []

    for (x1, y1, x2, y2, label, box_color) in pose_boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)

        ty = int(y1) - 10
        while any(abs(ty - uy) < 20 for uy in used_y):
            ty -= 20

        used_y.append(ty)

        cv2.rectangle(img, (int(x1), ty - 18),
                      (int(x1) + len(label) * 9, ty + 5),
                      box_color, -1)

        cv2.putText(img, label, (int(x1), ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)

    return img, used_y


# =========================================================
# Box overlay
# =========================================================
def draw_box_on_image(img, r_box, names, pose_used_y):

    if r_box.boxes is None:
        return img

    box_entries = []
    for box in r_box.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = names[cls].lower()

        label = f"{names[cls]} {conf:.2f}"
        box_entries.append((x1, y1, x2, y2, label, cls_name))

    box_entries = sorted(box_entries, key=lambda x: x[1])
    used_y = pose_used_y.copy()

    for (x1, y1, x2, y2, label, cls_name) in box_entries:

        if cls_name == "violence":
            box_color = (0, 0, 255)
        elif cls_name == "nonviolence":
            box_color = (255, 128, 0)
        else:
            box_color = (200, 200, 200)

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)

        ty = int(y1) - 10
        while any(abs(ty - uy) < 20 for uy in used_y):
            ty -= 20

        used_y.append(ty)

        cv2.rectangle(img, (int(x1), ty - 18),
                      (int(x1) + len(label) * 9, ty + 5),
                      box_color, -1)

        cv2.putText(img, label, (int(x1), ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return img


# =========================================================
# 영상 전체 분석
# =========================================================
def analyze_video(video_path, yolo_pose, yolo_box, transformer, original_name):

    import json
    import cv2
    import numpy as np
    import torch
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---------------------------------------------------------
    # 🔥 영상 해상도 절반으로 줄이기 (용량 1/4~1/6로 감소)
    # ---------------------------------------------------------
    resize_w = int(w * 0.5)
    resize_h = int(h * 0.5)

    # ---------------------------------------------------------
    # 🔥 저장 경로
    # ---------------------------------------------------------
    PREVIEW_DIR = PROJECT_ROOT / "results" / "preview"
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    name_stem = Path(original_name).stem
    save_name = PREVIEW_DIR / f"preview_{name_stem}.mp4"

    # ---------------------------------------------------------
    # 🔥 H.264(avc1) 코덱 사용 → 용량 줄고 브라우저 호환성 높음
    # ---------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(save_name), fourcc, fps, (resize_w, resize_h))

    # 위험도 기록 저장 리스트
    risk_log = []

    LABEL_MAP = ["Safety", "Warning", "Danger"]
    COLOR_MAP = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]

    tf_label, tf_score = "Safety", 0.0

    progress = st.progress(0)
    status_text = st.empty()

    seq_buffer = []
    frame_idx = 0

    # ---------------------------------------------------------
    # 🔥 안정적인 writer 종료를 위한 try/finally
    # ---------------------------------------------------------
    try:
        while True:

            if st.session_state.get("stop", False):
                st.warning("🛑 분석 중지되었습니다.")
                break

            ret, frame = cap.read()
            if not ret:
                break

            # ---------------- YOLO Pose ----------------
            r_pose = yolo_pose(frame, conf=0.7)[0]
            img = frame.copy()
            img, pose_used_y = draw_pose_on_image(img, r_pose, yolo_pose.names)

            # ---------------- YOLO Box ----------------
            r_box = yolo_box(frame, conf=0.5)[0]
            img = draw_box_on_image(img, r_box, yolo_box.names, pose_used_y)

            # ---------------- Transformer ----------------
            pose_vec = extract_pose_vector(r_pose)
            if pose_vec is not None:
                seq_buffer.append(pose_vec)

            if len(seq_buffer) >= 12:
                seq = np.array(seq_buffer[-12:], dtype=np.float32)
                seq_tensor = torch.tensor(seq).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = transformer(seq_tensor)
                    prob = torch.softmax(out, dim=1)[0].cpu().numpy()

                pred = int(np.argmax(prob))
                tf_label = LABEL_MAP[pred]
                tf_score = float(prob[pred])

            # 🔥 화면 표시용 글자
            tf_color = COLOR_MAP[LABEL_MAP.index(tf_label)]
            cv2.putText(img, f"{tf_label} {tf_score:.2f}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, tf_color, 3, cv2.LINE_AA)

            # 🔥 여기서 해상도 줄여서 저장 (핵심)
            img_small = cv2.resize(img, (resize_w, resize_h))
            writer.write(img_small)

            # 위험도 로그 저장
            risk_log.append({
                "frame": frame_idx,
                "time": frame_idx / fps,
                "risk": float(tf_score),
                "label": tf_label
            })

            progress.progress((frame_idx + 1) / total_frames)
            status_text.text(f"📊 분석 중... {frame_idx+1} / {total_frames}")

            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    # ---------------------------------------------------------
    # 🔥 위험도 JSON 저장
    # ---------------------------------------------------------
    RISK_DIR = PROJECT_ROOT / "results" / "risk"
    RISK_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RISK_DIR / f"{name_stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(risk_log, f, indent=4, ensure_ascii=False)

    progress.progress(1.0)
    st.success(f"🎉 분석 완료! 저장된 영상: {save_name}")
    st.info(f"💾 위험도 JSON 저장: {json_path}")


# =========================================================
# 프레임 미리보기 + 라벨링
# =========================================================
def show_frame_labeling_tab(selected_video):

    st.header("")

    # ------------------------------
    # 총 프레임 수 읽기
    # ------------------------------
    cap = cv2.VideoCapture(str(selected_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    st.info(f"총 프레임 수: **{total_frames}**")

    # ------------------------------
    # 이미지 크기 조절 슬라이더
    # ------------------------------
    img_width = st.slider(
        "🖼️ 이미지 크기 ",
        min_value=300,
        max_value=1500,
        value=900,
        step=50,
        key="image_size_slider"
    )

    # --------------------------------------------------------
    # 🔥 1) 먼저 프레임을 세션에서 읽기 (없으면 0)
    # --------------------------------------------------------
    frame_idx = st.session_state.get("preview_frame_idx", 0)

    # ------------------------------
    # 선택 프레임 로드
    # ------------------------------
    cap = cv2.VideoCapture(str(selected_video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    # =============================
    # 🔥 단 한 번만 이미지 출력 (위쪽)
    # =============================
    if ret:
        st.image(frame, channels="BGR", caption=f"Frame {frame_idx}", width=img_width)


    # ============================================================
    # 🔥 2) 프레임 선택 슬라이더 (이미지 아래에 위치)
    # ============================================================
    new_frame_idx = st.slider(
        "🎞 프레임 선택",
        min_value=0,
        max_value=total_frames - 1,
        value=frame_idx,
        key="preview_frame_slider"
    )

    # 프레임이 바뀌면 화면 전체 새로고침
    if new_frame_idx != frame_idx:
        st.session_state.preview_frame_idx = new_frame_idx
        st.rerun()

    st.markdown("---")

    # ============================================================
    # 🔥 선택한 프레임 분석 결과
    # ============================================================
    st.subheader(" ")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    @st.cache_resource
    def load_models():
        yolo_pose = YOLO("models/best.pt")
        yolo_box = YOLO("models/Lisen.pt")
        transformer = TransformerClassifier(input_dim=34, num_classes=3)

        model_path = (
            Path("models/transformer_action_pose.pt")
            if Path("models/transformer_action_pose.pt").exists()
            else Path("models/transformer_action_risk.pt")
        )

        transformer.load_state_dict(torch.load(model_path, map_location=device))
        transformer.to(device)
        transformer.eval()

        return yolo_pose, yolo_box, transformer

    yolo_pose, yolo_box, transformer = load_models()

    # YOLO Pose
    r_pose = yolo_pose(frame, conf=0.7)[0]
    adult = None
    child = None
    if r_pose.boxes is not None:
        for box in r_pose.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = yolo_pose.names[cls].lower()
            if name == "adult":
                adult = conf
            if name == "child":
                child = conf

    # YOLO Box
    r_box = yolo_box(frame, conf=0.5)[0]
    violence = None
    nonviolence = None
    if r_box.boxes is not None:
        for box in r_box.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = yolo_box.names[cls].lower()
            if name == "violence":
                violence = conf
            if name == "nonviolence":
                nonviolence = conf

    # Transformer
    pose_vec = extract_pose_vector(r_pose)
    tf_label = "N/A"
    tf_score = 0.0

    if pose_vec is not None:
        seq = np.array([pose_vec] * 12, dtype=np.float32)
        seq = torch.tensor(seq).unsqueeze(0).to(device)

        with torch.no_grad():
            out = transformer(seq)
            prob = torch.softmax(out, dim=1)[0].cpu().numpy()

        label_map = ["Safety", "Warning", "Danger"]
        pred = int(np.argmax(prob))
        tf_label = label_map[pred]
        tf_score = float(prob[pred])
    # =============================
    # ✨ 카드형 UI 컴포넌트
    # =============================
    def metric_card(title, value, color="#4CAF50"):
        st.markdown(
            f"""
            <div style="
                padding:15px;
                border-radius:10px;
                border:1px solid #ddd;
                background:#ffffff;
                margin-bottom:10px;
                box-shadow:0px 2px 4px rgba(0,0,0,0.08);
            ">
                <div style="font-size:16px; font-weight:600; margin-bottom:5px;">
                    {title}
                </div>
                <div style="font-size:24px; font-weight:700; color:{color};">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

   # =========================================================
    # 🔥 선택한 프레임 분석 결과 (카드형 UI 버전)
    # =========================================================
    st.markdown("## 📊 프레임별 데이터")

    # -----------------------------------
    # 🧍 best.pt
    # -----------------------------------
    st.markdown("### 🧍 best")

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Detected Persons", len(r_pose.boxes) if r_pose.boxes is not None else 0)
        metric_card("Keypoints Extracted", "Yes" if pose_vec is not None else "No", "#2196F3")

    with col2:
        metric_card("Adult Confidence", round(adult, 3) if adult else "N/A", "#4CAF50")
        metric_card("Vector Length", len(pose_vec) if pose_vec is not None else 0, "#795548")

    with col3:
        metric_card("Child Confidence", round(child, 3) if child else "N/A", "#E91E63")


    # -----------------------------------
    # 🟥 Lisen.pt
    # -----------------------------------
    st.markdown("### 🟥 Lisen")

    col4, col5, col6 = st.columns(3)

    with col4:
        metric_card("Detected Objects", len(r_box.boxes) if r_box.boxes is not None else 0)
        metric_card("Violence Boxes", sum(1 for b in r_box.boxes if yolo_box.names[int(b.cls[0])].lower() == "violence") if r_box.boxes is not None else 0, "#F44336")

    with col5:
        metric_card("Violence Score", round(violence, 3) if violence else "N/A", "#FF5722")

    with col6:
        metric_card("Non-Violence Score", round(nonviolence, 3) if nonviolence else "N/A", "#03A9F4")


    # -----------------------------------
    # 🔮 transformer_action_risk.pt
    # -----------------------------------
    st.markdown("### 🔮 Transformer")

    emoji = "🟢" if tf_label == "Safety" else "🟡" if tf_label == "Warning" else "🔴"
    risk_color = "#4CAF50" if tf_label == "Safety" else "#FFC107" if tf_label == "Warning" else "#F44336"

    col7, col8, col9 = st.columns(3)

    with col7:
        metric_card("Predicted Label", f"{emoji} {tf_label}", risk_color)

    with col8:
        metric_card("Risk Score", round(tf_score, 3), risk_color)

    with col9:
        metric_card("Sequence Length", 12, "#795548")

    # 추가 Softmax 시각화
    if pose_vec is not None:
        st.markdown("#### 📊 Transformer Softmax ")
        st.bar_chart({
            "Safety": prob[0],
            "Warning": prob[1],
            "Danger": prob[2]
        })

# =========================================================
# 모델 추론 탭 (업로드 하나로 통일)
# =========================================================
def show_model_inference_tab(MODEL_DIR, DATA_PROC):

    st.header("🎬 모델 추론 ")

    if "stop" not in st.session_state:
        st.session_state.stop = False

    # ----------------------------------------------------------------------
    # ▶ 파일 업로드
    # ----------------------------------------------------------------------
    uploaded_file = st.file_uploader("📤 분석할 mp4 영상 업로드", type=["mp4"])

    if uploaded_file is None:
        st.info("📂 분석하려면 영상을 업로드하세요.")
        return None

    # ✔ 원본 파일명 저장 (preview 저장 시 매우 중요!)
    st.session_state["original_video_name"] = uploaded_file.name

    # 파일 메모리에 저장
    video_bytes = uploaded_file.read()
    st.session_state.video_memory = video_bytes

    st.success(f"업로드 완료: {uploaded_file.name}")
    st.markdown("---")

    # ----------------------------------------------------------------------
    # ▶ 임시파일로 변환 (모델 분석용)
    # ----------------------------------------------------------------------
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(st.session_state.video_memory)
    temp_video.flush()

    selected_video = temp_video.name  # 이 경로만 YOLO/OpenCV에 전달

    # ----------------------------------------------------------------------
    # ▶ 버튼 UI
    # ----------------------------------------------------------------------
    col_start, col_space, col_stop = st.columns([3, 4, 3])

    with col_start:
        start_clicked = st.button("🚀 분석 시작", use_container_width=True)

    with col_stop:
        stop_clicked = st.button("■ 분석 정지", use_container_width=True)

    if stop_clicked:
        st.session_state.stop = True

    # ----------------------------------------------------------------------
    # ▶ 모델 로드
    # ----------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo_pose = YOLO(str(MODEL_DIR / "best.pt"))
    yolo_box = YOLO(str(MODEL_DIR / "Lisen.pt"))

    transformer = TransformerClassifier(input_dim=34, num_classes=3)
    transformer.load_state_dict(
        torch.load(MODEL_DIR / "transformer_action_pose.pt", map_location=device)
        if (MODEL_DIR / "transformer_action_pose.pt").exists()
        else torch.load(MODEL_DIR / "transformer_action_risk.pt", map_location=device)
    )
    transformer.to(device)
    transformer.eval()

    # ----------------------------------------------------------------------
    # ▶ 분석 실행
    # ----------------------------------------------------------------------
    if start_clicked:
        st.session_state.stop = False
        with st.spinner("⏳ 영상 분석 중입니다..."):
            analyze_video(
                selected_video, 
                yolo_pose, 
                yolo_box, 
                transformer,
                st.session_state["original_video_name"]  # 🔥 중요
            )

    return selected_video