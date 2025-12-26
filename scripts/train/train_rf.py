from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

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


def train_random_forest(
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
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    n_jobs: int,
    seed: int,
):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth and max_depth > 0 else None,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=seed,
    )
    rf.fit(prep.X_train, enc.y_train)
    probs_train = rf.predict_proba(prep.X_train)
    probs_test = rf.predict_proba(prep.X_test)
    preds_test = probs_test.argmax(axis=1)
    acc = accuracy_score(enc.y_test, preds_test)
    ll_train = log_loss(enc.y_train, probs_train)
    ll_test = log_loss(enc.y_test, probs_test)

    plot_loss_curve([ll_train], [ll_test], out_path=loss_path, ylabel="Log-Loss")

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        return rf.predict(arr)

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

    print(f"[RF] Test-Accuracy: {acc:.4f} | Train-LogLoss={ll_train:.4f} Test-LogLoss={ll_test:.4f}")
    print(classification_report(enc.y_test, preds_test, target_names=class_names, digits=4))
    print("Confusion matrix:\n", confusion_matrix(enc.y_test, preds_test))
    plot_confusion_matrix(enc.y_test, preds_test, class_names=class_names, out_path=cm_path)

    class_centers = compute_class_centers(prep.X_train, enc.y_train, len(class_names))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "model": rf,
        "preprocessor": prep.preprocessor,
        "feature_cols": feature_cols,
        "class_names": class_names,
        "col_bounds": prep.col_bounds,
        "class_centers": class_centers,
    }
    joblib.dump(payload, model_path)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"class_names": class_names}, fh, ensure_ascii=False, indent=2)
    print(f"RandomForest-Artefakte gespeichert: {model_path}, {meta_path}")


def main():
    ap = argparse.ArgumentParser(description="RandomForest-Training für LymphDot")
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

    ap.add_argument("--rf-n-estimators", dest="rf_n_estimators", type=int, default=300)
    ap.add_argument("--rf-max-depth", dest="rf_max_depth", type=int, default=0, help="0 oder <=0 bedeutet None")
    ap.add_argument("--rf-min-samples-leaf", dest="rf_min_samples_leaf", type=int, default=1)
    ap.add_argument("--rf-n-jobs", dest="rf_n_jobs", type=int, default=-1)
    ap.add_argument("--rf-model", dest="rf_model", type=str, default="outputs/rf/model.joblib")
    ap.add_argument("--rf-meta", dest="rf_meta", type=str, default="outputs/rf/meta.json")
    ap.add_argument("--rf-plot", dest="rf_plot", type=str, default="outputs/rf/pca_regions.png")
    ap.add_argument("--rf-plot-3d", dest="rf_plot_3d", type=str, default="outputs/rf/pca_3d.html")
    ap.add_argument("--rf-loss-plot", dest="rf_loss_plot", type=str, default="outputs/rf/loss.png")
    ap.add_argument("--rf-cm-plot", dest="rf_cm_plot", type=str, default="outputs/rf/confusion_matrix.png")
    args = ap.parse_args()

    set_seed(args.seed)
    feature_cols, enc, prep = prepare_data(args.data, args.target, args.feature, args.test_size, args.seed)
    class_names = enc.class_names

    train_random_forest(
        feature_cols=feature_cols,
        enc=enc,
        prep=prep,
        class_names=class_names,
        model_path=Path(args.rf_model),
        meta_path=Path(args.rf_meta),
        plot_regions_path=Path(args.rf_plot),
        plot_3d_path=Path(args.rf_plot_3d),
        loss_path=Path(args.rf_loss_plot),
        cm_path=Path(args.rf_cm_plot),
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
        n_jobs=args.rf_n_jobs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
