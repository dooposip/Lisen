import streamlit as st
import platform, psutil, torch, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def render_sidebar():

    # ---------------------------
    # CSS
    # ---------------------------
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        min-width: 300px !important;
        max-width: 300px !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0 !important;
        max-width: 0 !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }
    .streamlit-expanderHeader {
        font-size: 13px !important;
        padding-top: 2px !important;
        padding-bottom: 2px !important;
    }
    .sidebar-small-text {
        font-size: 13px !important;
        line-height: 1.4 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------------------------
    # Sidebar Layout
    # ---------------------------
    with st.sidebar:

        # 🧬 Guide
        guide = st.expander("🧬 LiSEN 가이드", expanded=False)
        guide.markdown("""
        <div class="sidebar-small-text">
        <b>1️⃣~9️⃣ 순서대로 진행</b><br>
        <b>1️⃣ Pose 라벨링</b><br>
        YOLO Pose 자동 라벨 생성<br>
        <b>2️⃣ 데이터 검증</b><br>
        라벨 매칭 및 오류 점검<br>
        <b>3️⃣ YOLO 학습</b><br>
        Adult/Child 분류 모델 학습<br>
        <b>4️⃣ 행동 라벨링</b><br>
        Frame 행동 라벨 입력<br>
        <b>5️⃣ 시퀀스 생성</b><br>
        Pose → Sequence 변환<br>
        <b>6️⃣ 데이터 증강</b><br>
        시퀀스 증강<br>
        <b>7️⃣ Transformer 학습</b><br>
        위험도 학습<br>
        <b>8️⃣ 모델 추론</b><br>
        YOLO + Transformer 분석<br>
        <b>9️⃣ 실시간 위험 감지</b><br>
        실시간 위험 신호 감지<br>
        </div>
        """, unsafe_allow_html=True)

        # 💻 PC 상태
        pc_info = st.expander("💻 PC 상태", expanded=False)
        pc_info.markdown(f"""
        <div class='sidebar-small-text'>
        <b>🖥 OS:</b> {platform.system()} {platform.release()}<br>
        <b>⚙ CPU:</b> {platform.processor()}<br>
        <b>💾 RAM:</b> {round(psutil.virtual_memory().total / (1024**3), 1)} GB<br>
        <b>🐍 Python:</b> {platform.python_version()}<br>
        <b>🧱 Machine:</b> {platform.machine()}<br>
        </div>
        """, unsafe_allow_html=True)

        if torch.cuda.is_available():
            pc_info.success("🟢 GPU 사용 가능")
        else:
            pc_info.warning("🟡 GPU 없음 (CPU 모드)")

        # ---------------------------
        # 📊 최근 위험 분석 기록 (HTML 안전 버전)
        # ---------------------------
        import json
        from pathlib import Path
        from datetime import datetime

        risk_folder = PROJECT_ROOT / "results" / "risk"
        recent_log = st.expander("📊 최근 위험 신호 ", expanded=False)

        # JSON 파일 탐색
        risk_files = sorted(
            risk_folder.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if not risk_files:
            recent_log.markdown("<div class='sidebar-small-text'>아직 분석된 기록이 없습니다.</div>",
                                unsafe_allow_html=True)
        else:
            latest_file = risk_files[0]

            try:
                # ★ 파일명 → 영상 이름 자동 추출
                video_name = latest_file.stem + ".mp4"

                # ★ 파일 수정 시간 = 분석 날짜·시간
                mod_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
                timestamp = mod_time.strftime("%Y-%m-%d %H:%M:%S")

                with open(latest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Safety 제외
                danger_list = [item for item in data if item["label"] != "Safety"]
                danger_list = danger_list[-5:][::-1]

                if not danger_list:
                    recent_log.markdown("<div class='sidebar-small-text'>위험 감지 기록이 없습니다.</div>",
                                        unsafe_allow_html=True)
                else:
                    html_blocks = []

                    for item in danger_list:
                        block = (
                            "<div style='margin-bottom:10px; padding:10px; border-radius:8px; "
                            "background:#ffffff; border:1px solid #ddd;'>"
                            
                            # 날짜/시간
                            f"<span style='font-size:12px; color:#777;'>📅 {timestamp}</span><br>"

                            # Frame, 시간, 위험도
                            f"<b>⏱ Frame {item['frame']}</b><br>"
                            f"▶ 시간: {item['time']:.2f}초<br>"
                            f"▶ 위험도: <span style='color:#d9534f;'>{item['label']}</span>"

                            # 영상 이름
                            f"<span style='font-size:12px; color:#555;'>🎬 {video_name}</span><br><br>"
                            "</div>"
                        )
                        html_blocks.append(block)

                    final_html = "<div class='sidebar-small-text'>" + "".join(html_blocks) + "</div>"
                    recent_log.markdown(final_html, unsafe_allow_html=True)

            except Exception as e:
                recent_log.error(f"오류 발생: {e}")