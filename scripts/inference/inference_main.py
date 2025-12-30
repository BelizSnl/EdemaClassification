from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse helpers from the existing inference modules
from scripts.inference.inference_nn import (  # type: ignore
    apply_bounds,
    ensure_columns,
    load_artifacts as load_nn_artifacts,
    normalize_gender,
    get_device,
)
from scripts.inference.inference_svm import load_svm_artifacts  # type: ignore
from scripts.inference.inference_rf import load_rf_artifacts  # type: ignore


def _prepare_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    preprocessor,
    col_bounds: Dict[str, Tuple[float, float]] | None = None,
) -> np.ndarray:
    """Align columns, normalize gender and apply clipping before transform."""
    x_df = ensure_columns(df, feature_cols)
    x_df = normalize_gender(x_df)
    x_df = apply_bounds(x_df, col_bounds or {})
    return preprocessor.transform(x_df).astype("float32")


class EnsembleInference:
    def __init__(
        self,
        nn_model: str = "outputs/nn/model.pt",
        nn_preproc: str = "outputs/nn/preprocessor.joblib",
        nn_meta: str = "outputs/nn/meta.json",
        svm_model: str = "outputs/svm/model.joblib",
        rf_model: str = "outputs/rf/model.joblib",
    ):
        self.device = get_device()
        self.nn_model, self.nn_preproc, self.nn_features, self.class_names, self.nn_bounds, _ = load_nn_artifacts(
            nn_model, nn_preproc, nn_meta
        )
        self.nn_model = self.nn_model.to(self.device).eval()

        self.svm_artifacts = load_svm_artifacts(svm_model)
        self.rf_artifacts = load_rf_artifacts(rf_model)

        self._validate_class_names()
        self.feature_cols = self.nn_features

    def _validate_class_names(self):
        names_svm = self.svm_artifacts["class_names"]
        names_rf = self.rf_artifacts["class_names"]
        if names_svm != self.class_names or names_rf != self.class_names:
            raise ValueError("Klassen-Reihenfolge von NN/SVM/RF ist nicht identisch.")

    def _predict_nn(self, df: pd.DataFrame) -> np.ndarray:
        X = _prepare_matrix(df, self.nn_features, self.nn_preproc, self.nn_bounds)
        xb = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.nn_model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def _predict_svm(self, df: pd.DataFrame) -> np.ndarray:
        art = self.svm_artifacts
        X = _prepare_matrix(df, art["feature_cols"], art["preprocessor"], art.get("col_bounds"))
        probs = art["model"].predict_proba(X)
        return probs

    def _predict_rf(self, df: pd.DataFrame) -> np.ndarray:
        art = self.rf_artifacts
        X = _prepare_matrix(df, art["feature_cols"], art["preprocessor"], art.get("col_bounds"))
        probs = art["model"].predict_proba(X)
        return probs

    def predict_dataframe(self, df: pd.DataFrame, topk: int = 3) -> Dict[str, object]:
        """Return per-model and averaged probabilities for a prepared DataFrame."""
        probs_nn = self._predict_nn(df)
        probs_svm = self._predict_svm(df)
        probs_rf = self._predict_rf(df)
        stacked = np.stack([probs_nn, probs_svm, probs_rf], axis=0)
        probs_avg = stacked.mean(axis=0)
        preds = probs_avg.argmax(axis=1)

        def _top_lines(idx: int) -> str:
            order = probs_avg[idx].argsort()[::-1][:topk]
            return ", ".join([f"{self.class_names[i]}={probs_avg[idx][i]:.3f}" for i in order])

        summary = [f"[{i}] pred={self.class_names[preds[i]]} | top{topk}: {_top_lines(i)}" for i in range(len(df))]
        return {
            "class_names": self.class_names,
            "avg_probs": probs_avg,
            "preds": preds,
            "per_model": {"nn": probs_nn, "svm": probs_svm, "rf": probs_rf},
            "summary": summary,
        }

    def predict_csv(self, csv_path: str, topk: int = 3) -> Dict[str, object]:
        df = pd.read_csv(csv_path)
        return self.predict_dataframe(df, topk=topk)


def main():
    ap = argparse.ArgumentParser(description="Soft-Voting Ensemble (NN + SVM + RF)")
    ap.add_argument("--csv", required=True, help="CSV mit neuen Fällen.")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--nn-model", dest="nn_model", default="outputs/nn/model.pt")
    ap.add_argument("--nn-preproc", dest="nn_preproc", default="outputs/nn/preprocessor.joblib")
    ap.add_argument("--nn-meta", dest="nn_meta", default="outputs/nn/meta.json")
    ap.add_argument("--svm-model", dest="svm_model", default="outputs/svm/model.joblib")
    ap.add_argument("--rf-model", dest="rf_model", default="outputs/rf/model.joblib")
    args = ap.parse_args()

    ensemble = EnsembleInference(
        nn_model=args.nn_model,
        nn_preproc=args.nn_preproc,
        nn_meta=args.nn_meta,
        svm_model=args.svm_model,
        rf_model=args.rf_model,
    )
    result = ensemble.predict_csv(args.csv, topk=args.topk)
    print("\n".join(result["summary"]))


if __name__ == "__main__":
    main()
