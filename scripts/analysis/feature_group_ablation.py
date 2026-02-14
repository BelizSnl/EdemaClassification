from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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


def _slugify(name: str) -> str:
    ascii_name = name.encode("ascii", "ignore").decode()
    base = ascii_name if ascii_name else name
    slug = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").lower()
    return slug or "scenario"


def _unique(seq: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(seq))


def _write_feature_config(path: Path, feature_cols: List[str], base_flags: Dict[str, bool], disabled: List[str]):
    ordered: Dict[str, bool] = {}
    for col in feature_cols:
        ordered[col] = bool(base_flags.get(col, True))
    for col in disabled:
        if col in ordered:
            ordered[col] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ordered, fh, ensure_ascii=False, indent=2)


def _normalize_confusion(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, np.clip(row_sums, a_min=1e-12, a_max=None))


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
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=0)
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)
    f1 = precision_recall_fscore_support(y_test, preds_test, average="macro", zero_division=0)[2]
    cm = confusion_matrix(y_test, preds_test, labels=list(range(n_classes)))
    return f1, _normalize_confusion(cm)


def _train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_train, y_train)
    preds_test = rf.predict(X_test)
    f1 = precision_recall_fscore_support(y_test, preds_test, average="macro", zero_division=0)[2]
    cm = confusion_matrix(y_test, preds_test, labels=list(range(n_classes)))
    return f1, _normalize_confusion(cm)


def main() -> int:
    ap = argparse.ArgumentParser(description="Group feature ablation for NN/SVM/RF.")
    ap.add_argument("--data", type=str, default="Lymphdoc_medi_4k.csv")
    ap.add_argument("--target", type=str, default="Klassifizierung")
    ap.add_argument("--feature-json", type=str, default="feature.json")
    ap.add_argument("--out-dir", type=str, default="outputs/feature-ablation/groups")
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
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    feature_dir = out_dir / "feature-json"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

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
    base_enabled = [c for c in all_features if bool(base_flags.get(c, True))]

    groups = {
        "group1": ["Ueber Brust", "Tallie cT", "H\u00fcfte cH"],
        "group2": ["Arm links cC", "Arm links cG", "Arm rechts cC", "Arm rechts cG"],
        "group3": ["Geschlecht", "Gr\u00f6\u00dfe"],
    }

    scenarios = [
        ("all_features_on", []),
        ("group1_off", groups["group1"]),
        ("group2_off", groups["group2"]),
        ("group3_off", groups["group3"]),
        ("group1_and_2_off", groups["group1"] + groups["group2"]),
        ("group2_and_3_off", groups["group2"] + groups["group3"]),
        ("group1_and_3_off", groups["group1"] + groups["group3"]),
        ("group1_and_2_and_3_off", groups["group1"] + groups["group2"] + groups["group3"]),
    ]

    results = []

    for name, disabled_raw in scenarios:
        disabled = _unique(disabled_raw)
        missing = [c for c in disabled if c not in all_features]
        if missing:
            print(f"Warnung: {name} enthält unbekannte Features: {missing}")
        disabled = [c for c in disabled if c in all_features]

        enabled_cols = [c for c in base_enabled if c not in disabled]
        if not enabled_cols:
            print(f"Überspringe {name}: keine aktiven Features übrig.")
            continue

        slug = _slugify(name)
        feature_path = feature_dir / f"feature_group__{slug}.json"
        _write_feature_config(feature_path, all_features, base_flags, disabled)

        print(f"\nScenario: {name}")
        print(f"Disabled: {disabled if disabled else '[none]'}")

        prep = scale_features(split.X_train[enabled_cols], split.X_test[enabled_cols])
        Xtr, Xte = prep.X_train, prep.X_test
        ytr, yte = enc_all.y_train, enc_all.y_test
        n_classes = len(class_names)

        f1_nn, _ = _train_nn(
            Xtr,
            ytr,
            Xte,
            yte,
            n_classes=n_classes,
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

        f1_svm, _ = _train_svm(Xtr, ytr, Xte, yte, n_classes=n_classes, seed=args.seed)
        f1_rf, _ = _train_rf(Xtr, ytr, Xte, yte, n_classes=n_classes, seed=args.seed)

        f1_avg = float(np.mean([f1_nn, f1_svm, f1_rf]))

        results.append(
            {
                "scenario": name,
                "disabled_features": "|".join(disabled) if disabled else "",
                "f1_nn": f1_nn,
                "f1_svm": f1_svm,
                "f1_rf": f1_rf,
                "f1_avg": f1_avg,
                "feature_json": str(feature_path),
            }
        )

    csv_path = out_dir / "group_ablation_f1.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("scenario,disabled_features,f1_nn,f1_svm,f1_rf,f1_avg,feature_json\n")
        for row in results:
            fh.write(
                f"{row['scenario']},"
                f"{row['disabled_features']},"
                f"{row['f1_nn']:.6f},"
                f"{row['f1_svm']:.6f},"
                f"{row['f1_rf']:.6f},"
                f"{row['f1_avg']:.6f},"
                f"{row['feature_json']}\n"
            )

    print(f"\nDone. Summary: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
