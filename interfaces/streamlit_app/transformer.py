import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import time


# --------------------------
# Dataset
# --------------------------
class PoseSeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# --------------------------
# Positional Encoding
# --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


# --------------------------
# Transformer Classifier
# --------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=2, dim_ff=256):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


# --------------------------
# Training Function (FINAL)
# --------------------------
def train_transformer_model(
        seq_path: Path,
        model_save_path: Path,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        batch_size=32,
        epochs=40,
        lr=1e-3,
        device=None,
        log_file: Path | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # 🔥 1) 증강된 시퀀스 파일 우선 사용
    # ============================================================
    seq_path = Path(seq_path)
    seq_dir = seq_path.parent

    seq_aug = seq_dir / "dataset_action_pose_aug.npz"
    seq_org = seq_dir / "dataset_action_pose.npz"

    if seq_aug.exists():
        print("📦 증강된 시퀀스 데이터셋 발견 → 자동 사용합니다.")
        seq_file = seq_aug
    else:
        print("📁 증강본 없음 → 원본 시퀀스 사용합니다.")
        seq_file = seq_org

    # ============================================================
    # 2) 파일 유효성 체크
    # ============================================================
    if not seq_file.exists():
        raise FileNotFoundError(f"❌ 시퀀스 파일이 없습니다: {seq_file}")

    print(f"👉 사용 중인 시퀀스 파일: {seq_file}")

    # ============================================================
    # 3) Load Data
    # ============================================================
    data = np.load(seq_file, allow_pickle=True)
    X, Y = data["X"], data["Y"]

    N, T, F = X.shape
    num_classes = int(np.max(Y) + 1)

    # --------------------------
    # Train/Val Split
    # --------------------------
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    train_loader = DataLoader(PoseSeqDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PoseSeqDataset(X_val, Y_val), batch_size=batch_size)

    # --------------------------
    # Model
    # --------------------------
    model = TransformerClassifier(F, num_classes, d_model, nhead, num_layers, dim_ff).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_acc = 0.0
    best_state = None

    # --------------------------
    # Log File Init
    # --------------------------
    log_f = None
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_file, "w", encoding="utf-8")
        log_f.write("epoch,train_acc,train_loss,val_acc,val_loss,time,logits\n")

    # --------------------------
    # Training Loop
    # --------------------------
    for ep in range(1, epochs + 1):
        start = time.time()

        # ---- Train ----
        model.train()
        tr_correct, tr_total = 0, 0
        tr_loss_sum = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()
            optimizer.step()

            tr_loss_sum += loss.item() * xb.size(0)
            tr_correct += (pred.argmax(1) == yb).sum().item()
            tr_total += xb.size(0)

        train_acc = tr_correct / tr_total
        train_loss = tr_loss_sum / tr_total

        # ---- Val ----
        model.eval()
        val_correct, val_total = 0, 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)

                pred = model(xb)
                loss = criterion(pred, yb)

                val_loss_sum += loss.item() * xb.size(0)
                val_correct += (pred.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        val_acc = val_correct / val_total
        val_loss = val_loss_sum / val_total
        elapsed = time.time() - start

        # ---- Logging ----
        if log_f:
            log_f.write(f"{ep},{train_acc},{train_loss},{val_acc},{val_loss},{elapsed},none\n")

        print(f"[{ep}/{epochs}] Train Acc={train_acc:.3f} Val Acc={val_acc:.3f}")

        # ---- Best Save ----
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()

    # Save best model
    if best_state:
        torch.save(best_state, model_save_path)

    if log_f:
        log_f.close()

    return best_acc