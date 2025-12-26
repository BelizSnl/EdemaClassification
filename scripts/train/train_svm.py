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
from modules.vis.plots import (
    plot_loss_curve,
    plot_pca_decision_regions,
    plot_pca_3d_scatter,
    plot_pca_3d_correctness,
    plot_confusion_matrix,
)


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


def compute_class_centers(X: np.ndarray, y: np.ndarray, n_classes: int) -> List[List[float]]:
    centers: List[List[float]] = []
    for idx in range(n_classes):
        mask = y == idx
        if not np.any(mask):
            centers.append([float("nan")] * X.shape[1])
        else:
            centers.append(np.mean(X[mask], axis=0).tolist())
    return centers


def train_svm(
    feature_cols: List[str],
    enc,
    prep,
    class_names: List[str],
    model_path: Path,
    meta_path: Path,
    plot_regions_path: Path,
    plot_3d_path: Path,
    loss_path: Path,
    cm_path: Path,
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
    plot_loss_curve([ll_train], [ll_test], out_path=loss_path, ylabel="Log-Loss")

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        return model.predict(arr)

    plot_pca_decision_regions(
        X_train=prep.X_train,
        X_test=prep.X_test,
        y_test=preds_test,
        class_names=class_names,
        predict_fn=predict_fn,
        out_path=plot_regions_path,
        cmap_background="Pastel2",
        cmap_points="tab10",
    )

    plot_pca_3d_scatter(
        X_train=prep.X_train,
        X_test=prep.X_test,
        labels=preds_test,
        class_names=class_names,
        out_path=plot_3d_path,
    )
    plot_pca_3d_correctness(
        X_train=prep.X_train,
        X_test=prep.X_test,
        y_true=enc.y_test,
        y_pred=preds_test,
        class_names=class_names,
        out_path=plot_3d_path.parent / f"{plot_3d_path.stem}_correctness{plot_3d_path.suffix}",
    )

    print(f"[SVM] Test-Accuracy: {acc:.4f} | Train-LogLoss={ll_train:.4f} Test-LogLoss={ll_test:.4f}")
    print(classification_report(enc.y_test, preds_test, target_names=class_names, digits=4))
    print("Confusion matrix:\n", confusion_matrix(enc.y_test, preds_test))
    plot_confusion_matrix(enc.y_test, preds_test, class_names=class_names, out_path=cm_path)

    class_centers = compute_class_centers(prep.X_train, enc.y_train, len(class_names))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "model": model,
        "preprocessor": prep.preprocessor,
        "feature_cols": feature_cols,
        "class_names": class_names,
        "col_bounds": prep.col_bounds,
        "class_centers": class_centers,
    }
    joblib.dump(payload, model_path)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"class_names": class_names}, fh, ensure_ascii=False, indent=2)
    print(f"SVM-Artefakte gespeichert: {model_path}, {meta_path}")


def main():
    ap = argparse.ArgumentParser(description="SVM-Training für LymphDot")
    ap.add_argument("--data", type=str, default="Lymphdoc_medi_4k.csv")
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
    ap.add_argument("--svm-plot", dest="svm_plot", type=str, default="outputs/svm/pca_regions.png")
    ap.add_argument("--svm-loss-plot", dest="svm_loss_plot", type=str, default="outputs/svm/loss.png")
    ap.add_argument("--svm-plot-3d", dest="svm_plot_3d", type=str, default="outputs/svm/pca_3d.html")
    ap.add_argument("--svm-cm-plot", dest="svm_cm_plot", type=str, default="outputs/svm/confusion_matrix.png")
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
        plot_regions_path=Path(args.svm_plot),
        plot_3d_path=Path(args.svm_plot_3d),
        loss_path=Path(args.svm_loss_plot),
        cm_path=Path(args.svm_cm_plot),
        kernel=args.svm_kernel,
        c_value=args.svm_c,
        gamma=args.svm_gamma,
    )


if __name__ == "__main__":
    main()
