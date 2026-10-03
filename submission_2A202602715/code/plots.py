"""plots.py — ảnh từng thí nghiệm (figures/<exp_id>.png) và ảnh chồng theo nhóm (figures/compare_<nhóm>.png)."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt

OPT_NAMES = {"sgd": "SGD", "sgd_momentum": "SGD+mom", "adam": "Adam", "adamw": "AdamW"}


def cfg_label(cfg: dict) -> str:
    """Tóm tắt cấu hình chính thành một dòng cho tiêu đề ảnh."""
    hidden = "-".join(str(h) for h in cfg["hidden"])
    clip = "none" if cfg["clip_norm"] is None else f"{cfg['clip_norm']:.3g}"
    return (f"{cfg['loss'].upper()} · {OPT_NAMES[cfg['optimizer']]} lr={cfg['lr']:g} wd={cfg['weight_decay']:g} · "
            f"batch={cfg['batch']} · {hidden} · drop={cfg['dropout']:g} · clip={clip} · "
            f"{cfg['precision']} · init={cfg['init']} · seed={cfg['seed']}")


def _maybe_log(ax, values):
    v = [x for x in values if x is not None and math.isfinite(x) and x > 0]
    if v and max(v) / min(v) > 50:
        ax.set_yscale("log")


def plot_run(result: dict, path: str) -> None:
    """Một thí nghiệm → PNG 3 ô: (1) train/val loss, (2) val acc + macro-F1, (3) grad_norm trước khi clip.

    Train loss đo ở chế độ eval trên tập con cố định của train. Đường đứt nét dọc = best_epoch.
    """
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = h["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    ax = axes[0]
    ax.plot(ep, h["train_loss"], "o-", ms=3, label="train loss (eval mode)")
    ax.plot(ep, h["val_loss"], "s-", ms=3, label="val loss")
    ax.axhline(s["step0_loss"], color="gray", ls=":", lw=1, label=f"bước 0 = {s['step0_loss']:.3f}")
    _maybe_log(ax, h["train_loss"] + h["val_loss"] + [s["step0_loss"]])
    ax.set(title=f"Loss ({cfg['loss'].upper()})", xlabel="epoch", ylabel="loss")

    ax = axes[1]
    ax.plot(ep, h["val_acc"], "o-", ms=3, label="val accuracy")
    ax.plot(ep, h["val_macro_f1"], "s-", ms=3, label="val macro-F1")
    ax.set(title=f"Val metrics (best: acc={s['val_acc']:.4f}, F1={s['val_macro_f1']:.4f})",
           xlabel="epoch", ylabel="giá trị")

    ax = axes[2]
    ax.plot(ep, h["grad_norm"], "o-", ms=3, label="grad_norm TB/epoch")
    ax.plot(ep, h["grad_norm_max"], "^--", ms=3, alpha=0.7, label="grad_norm max/epoch")
    if cfg["clip_norm"] is not None:
        ax.axhline(cfg["clip_norm"], color="red", ls=":", label=f"c = {cfg['clip_norm']:.3g} "
                                                                 f"(clip {100 * s['clip_frac']:.0f}% bước)")
    _maybe_log(ax, h["grad_norm"] + h["grad_norm_max"])
    ax.set(title="‖g‖₂ toàn cục (trước khi clip)", xlabel="epoch", ylabel="grad norm")

    for ax in axes:
        if s["best_epoch"]:
            ax.axvline(s["best_epoch"], color="green", ls="--", lw=1, alpha=0.6)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    title = f"{cfg['exp_id']}  —  {cfg_label(cfg)}"
    if s["diverged"]:
        title += "  [DIVERGED]"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric, path: str, title: str = "") -> None:
    """Vẽ chồng một hoặc nhiều chỉ số (str hoặc list[str]) của nhiều thí nghiệm, chú thích bằng exp_id."""
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.4), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        vals = []
        for r in results:
            ax.plot(r["history"]["epoch"], r["history"][m], "o-", ms=2.5, label=r["cfg"]["exp_id"])
            vals += r["history"][m]
        if m in ("grad_norm", "grad_norm_max", "train_loss", "val_loss"):
            _maybe_log(ax, vals)
        ax.set(title=m, xlabel="epoch", ylabel=m)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
