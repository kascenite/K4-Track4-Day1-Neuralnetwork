"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

N_CLASSES = 7
TRAIN_EVAL_SUBSET = 50_000   # train loss đo trên một tập con CỐ ĐỊNH của train (cùng cho mọi lần chạy)

# Cấu hình mặc định = BASELINE (M-base). `lr` được chọn bằng val trong notebook (Part 2).
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu mẫu số bằng 0 (như evaluate.py)."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Nhãn dự đoán int64 (N,) = argmax của logits, chế độ eval, FP32."""
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)])


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : F.cross_entropy trên logit thô và nhãn int64.
       "mse" : F.mse_loss(logits, one_hot(y)); reduction="mean" lấy trung bình trên MỌI phần tử B×7
               (không có hệ số 1/2), "sum" cộng hết rồi để người gọi chia.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        target = F.one_hot(y, N_CLASSES).to(logits.dtype)
        return F.mse_loss(logits, target, reduction=reduction)
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận {loss_name!r}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """dict(loss, acc, macro_f1) ở chế độ eval() và no_grad, FP32. Loss MSE chia cho N·7 (giống reduction="mean")."""
    model.eval()
    total_loss = torch.zeros((), device=X.device, dtype=torch.float64)
    cm = torch.zeros(N_CLASSES * N_CLASSES, device=X.device, dtype=torch.int64)
    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb).float()
        total_loss += compute_loss(logits, yb, loss_name, reduction="sum").double()
        cm += torch.bincount(yb * N_CLASSES + logits.argmax(dim=1), minlength=N_CLASSES ** 2)
    n = len(X) * (N_CLASSES if loss_name == "mse" else 1)
    cm = cm.view(N_CLASSES, N_CLASSES).cpu().numpy()
    return dict(loss=float(total_loss) / n, acc=float(np.trace(cm) / cm.sum()), macro_f1=macro_f1_from_confusion(cm))


def train_eval_subset(data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Tập con cố định ~50 000 mẫu train để đo train loss (generator riêng seed 0, độc lập với seed của lần chạy)."""
    if "train_eval_idx" not in data:
        g = torch.Generator().manual_seed(0)
        n = len(data["X_tr"])
        data["train_eval_idx"] = torch.randperm(n, generator=g)[:min(TRAIN_EVAL_SUBSET, n)].to(data["X_tr"].device)
    idx = data["train_eval_idx"]
    return data["X_tr"][idx], data["y_tr"][idx]


def build_model(cfg: dict, device) -> MLP:
    """Tạo model từ cfg và kiểm tra số tham số. `hidden` có thể là list (sau khi đọc JSON) nên ép về tuple."""
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden], f"số tham số sai: {count_params(model)}"
    return model


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về {"cfg", "history", "summary", "best_state"}.

    - Train loss đo ở chế độ eval trên tập con cố định của train, val loss/acc/macro-F1 trên toàn bộ val.
    - grad_norm: chuẩn L2 toàn cục TRƯỚC khi clip ở mỗi bước; ghi trung bình/max mỗi epoch và tỉ lệ bước
      có grad_norm > clip_norm (clipping thực sự kích hoạt). Các bước FP16 bị GradScaler bỏ qua (norm = inf)
      không tính vào trung bình.
    - Cộng dồn trên GPU, đồng bộ CPU một lần mỗi epoch (không .item() mỗi bước) để đo thời gian công bằng.
    - Dừng sớm và đặt diverged=True nếu train loss trung bình của epoch hoặc val loss là NaN/inf.
    - Không dùng X_eval ở bất kỳ đâu trong hàm này.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    assert cfg["lr"] is not None, "cfg['lr'] chưa được chọn"
    device = data["X_tr"].device
    set_seed(cfg["seed"])
    model = build_model(cfg, device)
    opt = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                          weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    precision = cfg["precision"]
    amp_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    scaler = torch.amp.GradScaler(device.type) if precision == "fp16" else None
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    X_sub, y_sub = train_eval_subset(data)
    clip = cfg["clip_norm"]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step0_loss = evaluate(model, X_val, y_val, cfg["loss"])["loss"]

    keys = ("epoch", "train_loss", "train_loss_running", "val_loss", "val_acc", "val_macro_f1",
            "grad_norm", "grad_norm_max", "clip_frac", "epoch_time_s")
    hist = {k: [] for k in keys}
    step_norms = []                      # mọi grad_norm theo bước (để chọn c cho clipping)
    best = dict(val_loss=math.inf, epoch=0, state=None, acc=None, f1=None)
    diverged = False

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        _sync(device)
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        norms = []
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], gen):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)     # luôn unscale trước khi đo/clip, nếu không norm bị nhân với scale
                norms.append(clip_gradients(model.parameters(), clip))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                norms.append(clip_gradients(model.parameters(), clip))
                opt.step()
            loss_sum += loss.detach().float()
        _sync(device)
        epoch_time = time.perf_counter() - t0

        norms = torch.stack(norms).float()
        finite = torch.isfinite(norms)
        fn = norms[finite]
        running = float(loss_sum) / len(norms)
        step_norms.append(fn.cpu())
        tr = evaluate(model, X_sub, y_sub, cfg["loss"])
        va = evaluate(model, X_val, y_val, cfg["loss"])
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr["loss"])
        hist["train_loss_running"].append(running)
        hist["val_loss"].append(va["loss"])
        hist["val_acc"].append(va["acc"])
        hist["val_macro_f1"].append(va["macro_f1"])
        hist["grad_norm"].append(float(fn.mean()) if len(fn) else float("nan"))
        hist["grad_norm_max"].append(float(fn.max()) if len(fn) else float("nan"))
        hist["clip_frac"].append(float((fn > clip).float().mean()) if clip is not None and len(fn) else 0.0)
        hist["epoch_time_s"].append(epoch_time)

        if va["loss"] < best["val_loss"]:
            best.update(val_loss=va["loss"], epoch=epoch, acc=va["acc"], f1=va["macro_f1"],
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if not (math.isfinite(running) and math.isfinite(va["loss"])):
            diverged = True
            break

    all_norms = torch.cat(step_norms) if step_norms else torch.zeros(0)
    q = (lambda p: float(torch.quantile(all_norms, p))) if len(all_norms) else (lambda p: float("nan"))
    summary = dict(
        step0_loss=step0_loss,
        best_val_loss=best["val_loss"] if best["epoch"] else float("nan"),
        best_epoch=best["epoch"],
        final_train_loss=hist["train_loss"][-1],
        final_val_loss=hist["val_loss"][-1],
        val_acc=best["acc"] if best["epoch"] else hist["val_acc"][-1],
        val_macro_f1=best["f1"] if best["epoch"] else hist["val_macro_f1"][-1],
        time_per_epoch_s=float(np.mean(hist["epoch_time_s"])),
        peak_mem_MB=torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else float("nan"),
        diverged=diverged,
        steps_per_epoch=len(norms),
        grad_norm_p50=q(0.5), grad_norm_p90=q(0.9), grad_norm_p99=q(0.99),
        grad_norm_max=float(all_norms.max()) if len(all_norms) else float("nan"),
        clip_frac=float(np.mean(hist["clip_frac"])),
        epochs_run=len(hist["epoch"]),
    )
    return {"cfg": cfg, "history": hist, "summary": summary, "best_state": best["state"]}


def _jsonable(cfg: dict) -> dict:
    return json.loads(json.dumps(cfg))


def run_cached(cfg: dict, data: dict, results_dir: str, ckpt_dir: str | None = None,
               force: bool = False, verbose: bool = True) -> dict:
    """Chạy run_experiment, hoặc nạp lại kết quả nếu results/<exp_id>.json đã có VỚI ĐÚNG cfg này.

    Dùng khi Colab ngắt kết nối giữa chừng: các lần chạy đã xong được nạp từ Drive thay vì huấn luyện lại.
    So sánh cả cfg (không chỉ exp_id), nên đổi lr/c... sẽ tự chạy lại. Nếu có ckpt_dir thì lưu/nạp best_state
    (file .pt nằm NGOÀI thư mục nộp).
    """
    from results_table import save_result

    cfg = _jsonable({**DEFAULT_CFG, **cfg})
    path = os.path.join(results_dir, f"{cfg['exp_id']}.json")
    ckpt = os.path.join(ckpt_dir, f"{cfg['exp_id']}.pt") if ckpt_dir else None
    if not force and os.path.exists(path):
        with open(path) as f:
            res = json.load(f)
        if res["cfg"] == cfg and (ckpt is None or os.path.exists(ckpt)):
            res["best_state"] = torch.load(ckpt, map_location="cpu") if ckpt else None
            if verbose:
                print(f"[cache] {cfg['exp_id']}: nạp lại {path}")
            return res
    res = run_experiment(cfg, data)
    res["cfg"] = cfg
    save_result(res, results_dir)
    if ckpt:
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(res["best_state"], ckpt)
    if verbose:
        s = res["summary"]
        print(f"[run] {cfg['exp_id']}: best_epoch={s['best_epoch']} val_loss={s['best_val_loss']:.4f} "
              f"val_acc={s['val_acc']:.4f} val_f1={s['val_macro_f1']:.4f} "
              f"t/epoch={s['time_per_epoch_s']:.2f}s diverged={s['diverged']}")
    return res


def write_predictions(row_id, preds, path: str) -> None:
    """CSV `row_id,pred` cho scripts/evaluate.py; đủ mọi dòng của eval, mỗi row_id đúng một lần."""
    row_id, preds = np.asarray(row_id), np.asarray(preds)
    assert len(row_id) == len(preds) and len(np.unique(row_id)) == len(row_id)
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("row_id,pred\n")
        f.writelines(f"{r},{p}\n" for r, p in zip(row_id.tolist(), preds.tolist()))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str,
               repo_root: str | None = None, out_json: str | None = None) -> dict | None:
    """Dùng cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval (FP32, eval mode), ghi CSV.

    Nếu có repo_root và out_json: chạy scripts/evaluate.py và trả về nội dung eval_result.json.
    """
    device = data["X_eval"].device
    model = build_model(cfg, device)
    model.load_state_dict(copy.deepcopy(result["best_state"]))
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    if repo_root is None or out_json is None:
        return None
    proc = subprocess.run([sys.executable, "scripts/evaluate.py", "--pred", os.path.abspath(pred_path),
                           "--out", os.path.abspath(out_json)],
                          cwd=repo_root, capture_output=True, text=True)
    print(proc.stdout, proc.stderr)
    proc.check_returncode()
    with open(out_json) as f:
        return json.load(f)
