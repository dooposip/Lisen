************************************요약***********************************
(0) 데이터 수집 + 자료
   ↓
(1) Pose 라벨링 → YOLO 학습용 데이터 생성
   ↓
(2) 검증 → label_quality.csv 생성
   ↓
(3) YOLO Adult/Child 학습 → best.pt 생성
   ↓
(4) 행동 라벨링 → *_action.csv 생성
   ↓
(5) 시퀀스 생성 → dataset_action_pose.npz 생성
   ↓
(6) 증강 → dataset_action_pose_aug.npz
   ↓
(7) Transformer 학습 → transformer_action_risk.pt
   ↓
(8) 추론 (테스트)
   ↓
(9) 실시간 위험 감지 (추론 실시간)


************************************상세************************************
🧬 LiSEN Dashboard 전체 파이프라인 — 탭별 사용 파일 & 역할
탭 순서가 머신러닝 워크플로우와 정확히 일치하도록 구성되어 있다.



✅ 0️⃣ 데이터 수집 (Raw Data)
파일 - > Helper.py
📌 사용하는 폴더 / 파일
data/raw/*.mp4
data/raw/*.json
data/raw/*.csv
📌 역할
원본 영상 및 참고자료를 탐색
어떤 파일들이 있는지 목록을 보여주고 구조 점검
아직 어떤 파일도 생성하지 않음 (읽기 전용)




✅ 1️⃣ Pose 라벨링 (Adult / Child + Keypoints 생성)
파일 -> label_tools.py
📌 입력
data/raw/*.mp4 (원본 영상)
📌 생성되는 파일
이미지
data/processed/images/train/*.jpg
data/processed/images/val/*.jpg
YOLO 라벨
data/processed/labels/train/*.txt
data/processed/labels/val/*.txt
📌 역할
YOLO Pose 모델이 Adult/Child 구분
사람 Keypoints(관절 위치)을 자동 라벨링해서 YOLO 학습용 데이터셋을 만든다.



✅ 2️⃣ 데이터 검증 (라벨 품질 검사)
파일 -> data_validation.py
📌 입력
data/processed/images/*
data/processed/labels/*
📌 생성되는 파일
data/processed/action_labels/csv
라벨 품질 점검 파일
잘못된 라벨, 누락된 라벨, 범위 오류 등을 자동 체크하며 생성됨
📌 역할
YOLO 학습 전 라벨 오류 검증
Adult/Child 잘못 라벨링된 부분 탐지
label_quality.csv는 학습용 데이터가 정상인지 점검하는 분석 리포트




✅ 3️⃣ YOLO Adult/Child 모델 학습
파일 -> training_tools.py
📌 입력
images/train/*
labels/train/*
images/val/*
labels/val/*
📌 생성되는 파일
models/best.pt 또는
results/pose_train/adult_child/weights/best.pt
📌 역할
Adult, Child, Keypoints까지
YOLO 기반으로 학습하는 단계.
이 모델은 이후:
시퀀스 생성
실시간 위험 감지 에서 사용됨.




✅ 4️⃣ 행동(Action) 라벨링
파일 -> action_labeler.py
📌 입력
data/raw/*.mp4
(사용자 수동 입력)
📌 생성되는 파일
Action CSV (구간 라벨링)
예: C012_A02_SY04_P01_S01_02DBS_action.csv
위치는: data/processed/action_labels/xxx_action.csv
📌 역할
사용자가 구간별(프레임 시작~끝)로 행동 라벨(0,1,2)을 수동 입력
Transformer 학습을 위한 “정답 레이블”을 만드는 단계
이 CSV가 Transformer 학습의 핵심 GT(Ground Truth)




✅ 5️⃣ 시퀀스 생성 (Action Pose Sequence)
파일 -> action_dataset_builder.py
📌 입력
YOLO 모델(best.pt)
행동 라벨 CSV (*_action.csv)
raw video
📌 생성되는 파일
시퀀스 데이터
data/processed/action_sequences/dataset_action_pose.npz
NPZ 내부
X: (N, 32, 34) Pose Sequence
Y: (N,) 행동 레이블(0/1/2)
📌 역할
YOLO keypoint + 행동 라벨을 매칭해
Transformer가 학습할 수 있는 시퀀스(X,Y)를 생성




✅ 6️⃣ 데이터 증강 (Augmentation)
파일 -> dataset_augmentation.py
📌 입력
dataset_action_pose.npz
📌 생성되는 파일
dataset_action_pose_aug.npz
📌 역할
시퀀스를 뒤집고(scale, jitter, shift)
Transformer 학습 다양화를 위한 증강 데이터 생성




✅ 7️⃣ Transformer 학습
파일 -> transformer.py
📌 입력
dataset_action_pose.npz
또는 dataset_action_pose_aug.npz
📌 생성되는 파일
data/processed/action_sequences/transformer_action_risk.pt
data/processed/action_sequences/transformer_training_log.txt
📌 역할
시퀀스 기반 위험 행동 분류 모델 학습
3단계 라벨(0,1,2)을 분류하는 Transformer 모델 생성




✅ 8️⃣ 모델 추론 (best + Lisen + Transformer )
관련 파일 -> info.py
📌 입력
YOLO 모델(best.pt, Lisen.pt )
Transformer 모델(transformer_action_risk.pt)
📌 생성 파일 
results/preview/preview_***mp4
results/risk/json
📌 역할
YOLO 단독 테스트
Transformer 단독 테스트
시퀀스 중 어떤 것이 위험인지 시각적 검증




✅ 9️⃣ 실시간 위험 감지 (최종 서비스 단계)
8번탭에서 분석 완료된 영상을 불러와서
작동 및 실시간 감지 기능구현 하려 했으나
실패 ...