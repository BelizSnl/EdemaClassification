from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import List, Dict

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
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
from modules.prep.feature import ensure_feature


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def prepare_data(
    data_path: str,
    target: str,
    feature_path: str | None,
    test_size: float,
    seed: int,
):
    df = load_data(data_path)
    split = split_dataset(df, target_col=target, test_size=test_size, random_state=seed)
    if feature_path:
        enabled_cols = ensure_feature(feature_path, split.feature_cols)
        split.X_train = split.X_train[enabled_cols]
        split.X_test = split.X_test[enabled_cols]
        split.feature_cols = enabled_cols
    name_to_idx = fit_label_encoder(split.y_train)
    enc = encode_labels(split.y_train, split.y_test, name_to_idx)
    prep = scale_features(split.X_train, split.X_test)
    return split.feature_cols, enc, prep


def train_svm(
    feature_cols: List[str],
    enc,
    prep,
    class_names: List[str],
    model_path: Path,
    meta_path: Path,
    kernel: str,
    c_value: float,
    gamma: str,
):
    model = SVC(kernel=kernel, C=c_value, gamma=gamma, probability=True, random_state=0)
    model.fit(prep.X_train, enc.y_train)

    probs_train = model.predict_proba(prep.X_train)
    probs_test = model.predict_proba(prep.X_test)
    preds_test = probs_test.argmax(axis=1)
    acc = accuracy_score(enc.y_test, preds_test)
    ll_train = log_loss(enc.y_train, probs_train)
    ll_test = log_loss(enc.y_test, probs_test)

    print(f"[SVM] Test-Accuracy: {acc:.4f} | Train-LogLoss={ll_train:.4f} Test-LogLoss={ll_test:.4f}")
    print(classification_report(enc.y_test, preds_test, target_names=class_names, digits=4))
    print("Confusion matrix:\n", confusion_matrix(enc.y_test, preds_test))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "model": model,
        "preprocessor": prep.preprocessor,
        "feature_cols": feature_cols,
        "class_names": class_names,
    }
    joblib.dump(payload, model_path)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"class_names": class_names}, fh, ensure_ascii=False, indent=2)
    print(f"SVM-Artefakte gespeichert: {model_path}, {meta_path}")


def main():
    ap = argparse.ArgumentParser(description="SVM-Training für LymphDot")
    ap.add_argument("--data", type=str, default="Lymphdoc_medi_gesammtdaten.csv")
    ap.add_argument("--target", type=str, default="Klassifizierung")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--feature",
        type=str,
        default="feature.json",
        help="Feature-Flag-Datei wie in train_nn.py (Spaltenname -> bool).",
    )

    ap.add_argument("--svm-kernel", dest="svm_kernel", type=str, default="rbf")
    ap.add_argument("--svm-c", dest="svm_c", type=float, default=1.0)
    ap.add_argument("--svm-gamma", dest="svm_gamma", type=str, default="scale")
    ap.add_argument("--svm-model", dest="svm_model", type=str, default="outputs/svm/model.joblib")
    ap.add_argument("--svm-meta", dest="svm_meta", type=str, default="outputs/svm/meta.json")
    args = ap.parse_args()

    set_seed(args.seed)
    feature_cols, enc, prep = prepare_data(args.data, args.target, args.feature, args.test_size, args.seed)
    class_names = enc.class_names

    train_svm(
        feature_cols=feature_cols,
        enc=enc,
        prep=prep,
        class_names=class_names,
        model_path=Path(args.svm_model),
        meta_path=Path(args.svm_meta),
        kernel=args.svm_kernel,
        c_value=args.svm_c,
        gamma=args.svm_gamma,
    )


if __name__ == "__main__":
    main()
