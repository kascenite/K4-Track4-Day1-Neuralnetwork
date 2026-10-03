"""data.py — nạp train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES, N_CLASSES = 54, 7


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz. Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id."""
    tr = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")
    X_train, y_train = tr["X"], tr["y"]
    X_eval, y_eval, row_id = ev["X"], ev["y"], ev["row_id"]
    for X, y in ((X_train, y_train), (X_eval, y_eval)):
        assert X.dtype == np.float32 and X.ndim == 2 and X.shape[1] == N_FEATURES
        assert y.dtype == np.int64 and y.shape == (len(X),)
        assert y.min() >= 0 and y.max() <= N_CLASSES - 1
    assert len(row_id) == len(X_eval)
    return X_train, y_train, X_eval, y_eval, row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval), phân tầng theo nhãn. Trả về X_tr, y_tr, X_val, y_val."""
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """mean/std của 10 cột số, CHỈ trên phần train còn lại.

    Tính trên val/eval (hoặc toàn bộ dữ liệu) là rò rỉ thông tin: thống kê của tập dùng để đánh giá
    lọt vào bước tiền xử lý, làm điểm val/eval lạc quan hơn thực tế.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)
    return num.mean(axis=0), num.std(axis=0)


def apply_standardizer(X, mean, std):
    """Bản sao của X với 10 cột đầu = (x - mean) / std; 44 cột nhị phân giữ nguyên. std = 0 thì giữ nguyên chia 1."""
    X = X.copy()
    safe_std = np.where(std > 0, std, 1.0)
    X[:, :N_NUMERIC] = ((X[:, :N_NUMERIC] - mean) / safe_std).astype(np.float32)
    return X


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader)."""
    X_full, y_full, X_eval, y_eval, row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)
    X_tr, X_val, X_eval = (apply_standardizer(X, mean, std) for X in (X_tr, X_val, X_eval))

    def t(a, dtype):
        return torch.tensor(a, dtype=dtype, device=device)

    data = dict(
        X_tr=t(X_tr, torch.float32), y_tr=t(y_tr, torch.int64),
        X_val=t(X_val, torch.float32), y_val=t(y_val, torch.int64),
        X_eval=t(X_eval, torch.float32), y_eval=t(y_eval, torch.int64),
        eval_row_id=row_id, mean=mean, std=std,
    )
    majority = np.bincount(y_tr, minlength=N_CLASSES).argmax()
    print(f"train {X_tr.shape}  val {X_val.shape}  eval {X_eval.shape}")
    print(f"'luôn đoán lớp đa số' (lớp {majority}) trên val: accuracy = {(y_val == majority).mean():.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Batch cuối nhỏ hơn batch_size vẫn được giữ lại (không bỏ mẫu nào); với 371 847 mẫu và batch 512
    batch cuối có 151 mẫu, ảnh hưởng không đáng kể.
    `generator` phải nằm trên cùng device với X (torch.Generator(device=X.device)).
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
