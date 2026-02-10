---
title: LiSEN - AI 기반 폭력 감지 시스템
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
---

<div style="text-align: center">
    <img src="https://capsule-render.vercel.app/api?type=waving&color=FFC2D1&height=120&text=LiSEN&animation=fadeIn&fontColor=000000&fontSize=60" style="width: 100%;" />
</div>

# LiSEN
- 실시간 폭력 감지 및 위험 평가 시스템
- Listen + Sense, “아이의 작은 신호를 듣다”

## 프로젝트 소개
어린이집의 가정 내 아동학대 의심 아동 조기 발견 및 신고 자동화 솔루션<br>
**실시간 비디오 스트림에서 폭력을 감지하고 위험을 평가하기 위한 고급 AI 대시보드**

## 팀원 소개
- [빙승현](https://github.com/ProjectBA0) : 자료 수집, 모델 학습 및 개발
- [정두균](https://github.com/dooposip) : streamlit 제작
- [지수정](https://github.com/ehqlsms1004) : PPT 제작 및 디자인
- [류주현](https://github.com/HyunRyuuu) : PL, 자료 수집, GITHUB ReadME 작성, 발표

## 개발기간 (11/03 ~ 12/05)
- **2025.11.03 ~ 2025.11.05** : 기획 / 데이터 수집 / 분석
- **2025.11.06 ~ 2025.11.13** : 데이터 수집 / 분석 / 전처리
- **2025.11.14 ~ 2025.11.28** : 모델 선택 / 모델 학습 / 모델 평가
- **2025.12.01 ~ 2025.12.05** : 웹서비스 구현 / 문서 정리

## 주요 기능
-   **실시간 모니터링:** YOLOv11 모델을 사용하여 폭력, 비폭력, 성인, 아동을 탐지합니다.
-   **포즈 분석:** 행동 분석을 위한 스켈레톤(골격) 키포인트를 시각화합니다.
-   **위험 평가:** 트랜스포머 모델을 사용하여 행동을 안전, 경고, 위험으로 분류합니다.
-   **대화형 대시보드:** 상세한 위험 지표를 통해 분석 결과를 프레임별로 검토할 수 있습니다.

<div style="text-align: left;">
    <h2 style="border-bottom: 1px solid #d8dee4; color: #282d33;"> 🛠️ Tech Stacks </h2> <br> 
    <div style="margin: ; text-align: left;" "text-align: left;">
        <img src="https://img.shields.io/badge/Github-181717?style=flat-square&logo=Github&logoColor=white">
        <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=Flask&logoColor=white">
    </div>
</div>
 
## 프로젝트 구조
이 문서는 LiSEN 프로젝트의 **현재 파일 및 폴더 구조**를 설명합니다.
```
LiSEN/
├── requirements.txt            # Python 패키지 의존성 목록.
├── README.md                   # 프로젝트 개요.
├── board.json                  # 게시판 데이터 저장 (JSON).
├── run_LiSEN_dashboard.bat     # Streamlit 대시보드 실행 스크립트 (Windows).
├── configs/
│   └── config.yaml             # 프로젝트 설정 파일 (모델 경로, YOLO/위험도 설정).
├── data/
│   ├── processed/              # 처리된 데이터 (라벨, 시퀀스 등).
│   │   └── pose_dataset.yaml   # YOLO Pose 데이터셋 설정 파일.
│   └── raw/                    # 원본 데이터 (비디오, JSON, CSV).
├── interfaces/
│   └── streamlit_app/          # Streamlit 웹 애플리케이션.
│       ├── app.py              # 앱 진입점 (메인 실행 파일).
│       ├── Helper.py           # 유틸리티 함수 (데이터 스캔, 요약 렌더링).
│       ├── label_tools.py      # Pose 라벨링 및 품질 검사 도구.
│       ├── training_tools.py   # YOLO 모델 학습 관련 도구.
│       ├── data_validation.py  # 데이터 유효성 검사 도구.
│       ├── action_labeler.py   # 행동 라벨링 인터페이스.
│       ├── action_dataset_builder.py # 행동 시퀀스 생성 도구.
│       ├── dataset_augmentation.py # 데이터 증강 도구.
│       ├── transformer.py      # Transformer 모델 정의 및 학습.
│       ├── info.py             # 모델 추론, 영상 분석 및 시각화.
│       └── sidebar.py          # Streamlit 사이드바 UI.
├── models/                     # AI 모델 가중치 파일.
│   └── yolo11m-pose.pt         # 기본 Pose 모델.
├── results/                    # (자동 생성) 분석 결과 저장 폴더 (preview, risk json 등).
```

## 시작하기
```conda
  # 가상환경 설정
  > conda create -n 가상환경이름 python=3.10
  > conda active 가상환경이름
```
```flask
  # cmd에서 입력
  # 인터프리터, 가상환경 설정 확인 이후 실행
  
  > streamlit run interfaces/streamlit_app/app.py
```
```flask
  # requirements 설치 필요 시
  > pip install -r requirements.txt
```

## 사용 방법
1.  **"Real-time Monitor (실시간 모니터링)"** 탭으로 이동합니다.
2.  비디오 파일을 업로드합니다.
3.  **"분석 시작"** 버튼을 클릭하여 AI 동작을 확인합니다.
4.  **"모델 추론 (Model Inference)"** 탭에서는 상세한 프레임별 라벨링 및 검증이 가능합니다.

## 사용된 모델
-   **Lisen.pt:** 포즈(Pose) 및 폭력/객체(Violence/Adult/Child) 감지를 위해 커스텀 학습된 YOLO 모델입니다.
-   **best_risk_transformer.pt:** 포즈 시퀀스를 기반으로 위험을 분류하는 트랜스포머 모델입니다.
-   **yolo11m-pose.pt:** (백업용) 기본 포즈 추정 모델입니다.

## 주요 파일 설명
-   **`core/realtime_processor.py`**: `1_Realtime_Monitor.py`에서 사용하는 실시간 영상 처리 엔진입니다. YOLO 모델과 Transformer 모델을 연동하여 프레임 단위로 분석하고 결과를 반환합니다.
-   **`interfaces/streamlit_app/pages/1_Realtime_Monitor.py`**: 사용자가 가장 먼저 접하는 **실시간 모니터링 대시보드**입니다. Altair 차트를 이용한 시각화와 사용자 태깅 기능이 구현되어 있습니다.
-   **`interfaces/streamlit_app/modules/info.py`**: 영상 분석(`analyze_video`), 결과 시각화(`draw_box_on_image`), 프레임 라벨링(`show_frame_labeling_tab`) 등 앱의 **핵심 기능**들이 구현된 모듈입니다.
-   **`interfaces/streamlit_app/tabs/tab8.py`**: `info.py`의 함수를 호출하여 **모델 추론 및 상세 분석 탭**을 렌더링합니다.
-   **`models/Lisen.pt`**: 이 프로젝트의 핵심 모델로, 사람의 관절(Pose)과 폭력 여부(Violence/Non-Violence), 객체 종류(Adult/Child)를 탐지하도록 학습되었습니다.
