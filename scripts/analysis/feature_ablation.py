from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.prep.data_prepare import (
    load_data,
    split_dataset,
    fit_label_encoder,
    encode_labels,
    scale_features,
)
from modules.nn.mlp import MLPClassifier
from modules.nn.utils import set_seed, get_device, make_dataloaders
from scripts.train.train_nn import train_one_epoch, evaluate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def _slugify(name: str) -> str:
    ascii_name = name.encode("ascii", "ignore").decode()
    base = ascii_name if ascii_name else name
    slug = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").lower()
    return slug or "feature"


def _write_feature_config(path: Path, feature_cols: List[str], base_flags: Dict[str, bool], disabled: str):
    ordered: Dict[str, bool] = {}
    for col in feature_cols:
        ordered[col] = bool(base_flags.get(col, True))
    if disabled in ordered:
        ordered[disabled] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ordered, fh, ensure_ascii=False, indent=2)


def _normalize_confusion(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, np.clip(row_sums, a_min=1e-12, a_max=None))


def _plot_confusion_avg(cm: np.ndarray, class_names: List[str], out_path: Path):
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Share", rotation=-90, va="bottom")

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Avg Confusion Matrix (normalized)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize="small",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _train_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    hidden: List[int],
    p_drop: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
    seed: int,
) -> tuple[float, np.ndarray]:
    set_seed(seed)
    device = get_device()
    train_loader, test_loader = make_dataloaders(
        X_train, y_train, X_test, y_test, batch_size=batch_size, device=device
    )

    model = MLPClassifier(X_train.shape[1], n_classes, hidden=tuple(hidden), p_drop=p_drop).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for _ in range(epochs):
        _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, _, _, _ = evaluate(model, test_loader, criterion, device)

        if te_loss < best_loss - early_stop_min_delta:
            best_loss = te_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if early_stop_patience and no_improve >= early_stop_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, _, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    f1 = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return f1, _normalize_confusion(cm)


def _train_svm(
    feature_cols: List[str],
    split,
    name_to_idx: Dict[str, int],
    seed: int,
) -> tuple[float, np.ndarray, List[str]]:
    class_names = [k for k, _ in sorted(name_to_idx.items(), key=lambda kv: kv[1])]
    enc = encode_labels(split.y_train, split.y_test, name_to_idx)
    prep = scale_features(split.X_train[feature_cols], split.X_test[feature_cols])

    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=0)
    model.fit(prep.X_train, enc.y_train)
    preds_test = model.predict(prep.X_test)
    f1 = precision_recall_fscore_support(enc.y_test, preds_test, average="macro", zero_division=0)[2]
    cm = confusion_matrix(enc.y_test, preds_test, labels=list(range(len(class_names))))
    return f1, _normalize_confusion(cm), class_names


def _train_rf(
    feature_cols: List[str],
    split,
    name_to_idx: Dict[str, int],
    seed: int,
) -> tuple[float, np.ndarray, List[str]]:
    class_names = [k for k, _ in sorted(name_to_idx.items(), key=lambda kv: kv[1])]
    enc = encode_labels(split.y_train, split.y_test, name_to_idx)
    prep = scale_features(split.X_train[feature_cols], split.X_test[feature_cols])

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(prep.X_train, enc.y_train)
    preds_test = rf.predict(prep.X_test)
    f1 = precision_recall_fscore_support(enc.y_test, preds_test, average="macro", zero_division=0)[2]
    cm = confusion_matrix(enc.y_test, preds_test, labels=list(range(len(class_names))))
    return f1, _normalize_confusion(cm), class_names


def main() -> int:
    ap = argparse.ArgumentParser(description="Single-feature ablation study for NN/SVM/RF.")
    ap.add_argument("--data", type=str, default="Lymphdoc_medi_4k.csv")
    ap.add_argument("--target", type=str, default="Klassifizierung")
    ap.add_argument("--feature-json", type=str, default="feature.json")
    ap.add_argument("--out-dir", type=str, default="outputs/feature-ablation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)

    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, nargs=2, default=[256, 128])
    ap.add_argument("--p-drop", type=float, default=0.1)
    ap.add_argument("--early-stop-patience", type=int, default=0)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0)
    ap.add_argument("--max-features", type=int, default=0, help="Limit number of features for quick tests (0=all).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    feature_dir = out_dir / "feature-json"
    cm_dir = out_dir / "confusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    split = split_dataset(df, target_col=args.target, test_size=args.test_size, random_state=args.seed)
    name_to_idx = fit_label_encoder(split.y_train)
    enc_all = encode_labels(split.y_train, split.y_test, name_to_idx)
    class_names = enc_all.class_names

    with open(args.feature_json, "r", encoding="utf-8") as fh:
        base_flags = json.load(fh)
    if not isinstance(base_flags, dict):
        raise ValueError("feature.json muss ein Objekt aus Spaltenname -> bool sein.")

    all_features = split.feature_cols
    ablation_features = [c for c in all_features if bool(base_flags.get(c, True))]
    if args.max_features and args.max_features > 0:
        ablation_features = ablation_features[: args.max_features]

    results = []

    for idx, disabled in enumerate(ablation_features, start=1):
        slug = f"{idx:02d}_{_slugify(disabled)}"
        feature_path = feature_dir / f"feature_off__{slug}.json"
        _write_feature_config(feature_path, all_features, base_flags, disabled)

        enabled_cols = [c for c in all_features if c != disabled and bool(base_flags.get(c, True))]
        if not enabled_cols:
            continue

        print(f"[{idx}/{len(ablation_features)}] Ablation: {disabled}")

        # NN
        prep_nn = scale_features(split.X_train[enabled_cols], split.X_test[enabled_cols])
        f1_nn, cm_nn = _train_nn(
            prep_nn.X_train,
            enc_all.y_train,
            prep_nn.X_test,
            enc_all.y_test,
            n_classes=len(class_names),
            hidden=args.hidden,
            p_drop=args.p_drop,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            seed=args.seed,
        )

        # SVM
        f1_svm, cm_svm, _ = _train_svm(enabled_cols, split, name_to_idx, args.seed)

        # RF
        f1_rf, cm_rf, _ = _train_rf(enabled_cols, split, name_to_idx, args.seed)

        f1_avg = float(np.mean([f1_nn, f1_svm, f1_rf]))
        cm_avg = (cm_nn + cm_svm + cm_rf) / 3.0

        cm_path = cm_dir / f"confusion_avg__{slug}.png"
        _plot_confusion_avg(cm_avg, class_names, cm_path)

        results.append(
            {
                "feature_disabled": disabled,
                "f1_nn": f1_nn,
                "f1_svm": f1_svm,
                "f1_rf": f1_rf,
                "f1_avg": f1_avg,
                "confusion_avg_png": str(cm_path),
                "feature_json": str(feature_path),
            }
        )

    # Save summary CSV
    csv_path = out_dir / "ablation_f1.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("feature_disabled,f1_nn,f1_svm,f1_rf,f1_avg,confusion_avg_png,feature_json\n")
        for row in results:
            fh.write(
                f"{row['feature_disabled']},"
                f"{row['f1_nn']:.6f},"
                f"{row['f1_svm']:.6f},"
                f"{row['f1_rf']:.6f},"
                f"{row['f1_avg']:.6f},"
                f"{row['confusion_avg_png']},"
                f"{row['feature_json']}\n"
            )

    print(f"\nDone. Summary: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
