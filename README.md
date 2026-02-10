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
프로젝트의 핵심 기능을 담당하는 파일들은 다음과 같습니다.

-   **`interfaces/streamlit_app/app.py`**: Streamlit 웹 애플리케이션의 메인 진입점. 전체 UI를 구성하고, 워크플로우 탭을 조율하며, 다른 모듈의 기능을 통합합니다.
-   **`interfaces/streamlit_app/Helper.py`**: 원본 데이터 스캔(`scan_raw_data`) 및 프로젝트 요약 정보(`render_project_summary`)를 UI에 렌더링하는 유틸리티 함수를 포함합니다.
-   **`interfaces/streamlit_app/label_tools.py`**: YOLO Pose 라벨 생성(`generate_yolo_pose_labels_stream`), 성인/아동 분류(`classify_adult_child`), 키포인트 추출 및 감지 결과 정리 등 데이터 준비의 핵심 로직을 담당합니다.
-   **`interfaces/streamlit_app/training_tools.py`**: YOLO 모델 학습(`yolo_pose_training_tab`)을 위한 UI와 로직을 구현하며, `pose_dataset.yaml`을 동적으로 생성합니다.
-   **`interfaces/streamlit_app/data_validation.py`**: 이미지-라벨 쌍(`validate_image_label_pairs`), YOLO 라벨 값(`validate_label_values`), 행동 라벨 CSV(`validate_action_labels`)의 유효성을 검증하여 데이터 품질을 보장합니다.
-   **`interfaces/streamlit_app/action_labeler.py`**: 사용자가 비디오 프레임에 수동으로 행동 라벨을 지정(`run_action_labeler`)할 수 있는 인터랙티브 도구를 제공합니다.
-   **`interfaces/streamlit_app/action_dataset_builder.py`**: YOLO 키포인트와 수동 행동 라벨을 결합하여 Transformer 모델 입력에 적합한 행동 포즈 시퀀스(`build_action_sequences`)를 구축합니다.
-   **`interfaces/streamlit_app/dataset_augmentation.py`**: 생성된 포즈 시퀀스에 지터, 스케일, 쉬프트, 좌우 반전 등의 데이터 증강(`apply_sequence_augmentations`)을 적용하여 모델의 견고성을 높입니다.
-   **`interfaces/streamlit_app/transformer.py`**: Transformer 모델의 아키텍처(`TransformerClassifier`) 및 학습(`train_transformer_model`) 절차를 정의하며, 위험 분류 시스템의 핵심 알고리즘입니다.
-   **`interfaces/streamlit_app/info.py`**: 모델 추론, 전체 비디오 분석(`analyze_video`), 결과 시각화, 상세 프레임별 분석(`show_frame_labeling_tab`) UI를 처리하며 YOLO 및 Transformer 모델을 통합합니다.
-   **`interfaces/streamlit_app/sidebar.py`**: Streamlit 사이드바의 콘텐츠를 렌더링하며, 프로젝트 가이드, PC 상태 정보, 최근 위험 분석 기록 등을 표시합니다.
-   **`config.yaml`**: YOLO 모델의 설정과 위험 평가 관련 매개변수를 정의하는 주요 설정 파일입니다.
-   **`requirements.txt`**: 프로젝트에 필요한 모든 Python 패키지 및 그 버전 정보를 명시합니다.
-   **`run_LiSEN_dashboard.bat`**: Windows 시스템에서 LiSEN 대시보드를 시작하기 위한 배치 스크립트입니다.
