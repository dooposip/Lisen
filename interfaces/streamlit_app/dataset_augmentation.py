# =========================================================
# Pose Sequence Augmentation (Stable Version for Transformer)
# =========================================================

from pathlib import Path
import numpy as np


# ---------------------------------------------------------
# 1) Dataset Load
# ---------------------------------------------------------
def load_sequence_dataset(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    return data["X"], data["Y"]


# ---------------------------------------------------------
# 2) Augmentation Functions (Safe for Pose)
# ---------------------------------------------------------

def add_jitter(seq, sigma=0.01):
    """약한 Gaussian jitter"""
    return seq + np.random.normal(0, sigma, seq.shape)


def scale_pose(seq, scale_range=(0.95, 1.05)):
    """포즈 전체 크기 스케일"""
    scale = np.random.uniform(*scale_range)
    return seq * scale


def temporal_shift(seq, max_shift=2):
    """시퀀스 플레임 시간축 이동"""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(seq, shift, axis=0)


# 좌우 관절 인덱스 교환 (17 keypoints, COCO 기준)
LEFT = [5, 7, 9, 11, 13, 15]
RIGHT = [6, 8, 10, 12, 14, 16]

def horizontal_flip(seq):
    """좌우 반전 pose 증강"""
    seq_flipped = seq.copy()

    # x좌표 반전
    seq_flipped[:, 0::2] = 1 - seq[:, 0::2]  

    # 좌우 관절 인덱스 swap
    for l, r in zip(LEFT, RIGHT):
        l_idx = l * 2
        r_idx = r * 2

        # (x,y) 세트를 통째로 swap
        left_xy = seq_flipped[:, l_idx:l_idx+2].copy()
        right_xy = seq_flipped[:, r_idx:r_idx+2].copy()

        seq_flipped[:, l_idx:l_idx+2] = right_xy
        seq_flipped[:, r_idx:r_idx+2] = left_xy

    return seq_flipped


# ---------------------------------------------------------
# 3) Master Augmentation Pipeline
# ---------------------------------------------------------
def augment_sequences(X, Y, aug_per_sample=2, 
                      use_flip=True,
                      use_jitter=True,
                      use_scale=True,
                      use_shift=True):

    X_aug = []
    Y_aug = []

    for seq, label in zip(X, Y):

        for _ in range(aug_per_sample):
            new_seq = seq.copy()

            if use_flip and np.random.rand() < 0.5:
                new_seq = horizontal_flip(new_seq)

            if use_jitter:
                new_seq = add_jitter(new_seq)

            if use_scale:
                new_seq = scale_pose(new_seq)

            if use_shift:
                new_seq = temporal_shift(new_seq)

            # 시퀀스 길이(T)는 항상 동일하므로 별도로 trim/pad 필요 없음
            X_aug.append(new_seq)
            Y_aug.append(label)

    return np.array(X_aug), np.array(Y_aug)


# ---------------------------------------------------------
# 4) Save
# ---------------------------------------------------------
def save_augmented_dataset(X, Y, save_path: Path):
    np.savez_compressed(save_path, X=X, Y=Y)
    return save_path

# ---------------------------------------------------------
# 5) Wrapper for Streamlit
# ---------------------------------------------------------
def apply_sequence_augmentations(X, Y,
                                 aug_count=2,
                                 use_flip=True,
                                 use_jitter=True,
                                 use_scale=True,
                                 use_shift=True):

    X_aug, Y_aug = augment_sequences(
        X,
        Y,
        aug_per_sample=aug_count,
        use_flip=use_flip,
        use_jitter=use_jitter,
        use_scale=use_scale,
        use_shift=use_shift
    )

    return X_aug, Y_aug