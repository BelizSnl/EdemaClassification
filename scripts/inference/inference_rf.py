from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ensure_columns(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    return df[feature_cols].copy()


def normalize_gender(df: pd.DataFrame) -> pd.DataFrame:
    if "Geschlecht" not in df.columns:
        return df
    male = {"m", "männlich", "maennlich", "mann", "male"}
    female = {"w", "weiblich", "frau", "female", "f"}

    def _map(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower().replace(",", ".")
        if s in male:
            return 0.0
        if s in female:
            return 1.0
        try:
            return float(s)
        except ValueError:
            return np.nan

    df["Geschlecht"] = df["Geschlecht"].apply(_map)
    return df


def apply_bounds(df: pd.DataFrame, bounds: Dict[str, Any]) -> pd.DataFrame:
    if not bounds:
        return df
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)
    return df


def detect_ood(X: np.ndarray, threshold: float) -> np.ndarray:
    """
    Markiert Zeilen als OOD, wenn der maximale |z|-Wert den Schwellwert übersteigt.
    threshold <= 0 deaktiviert die Warnung.
    """
    if threshold is None or threshold <= 0:
        return np.zeros(X.shape[0], dtype=bool)
    return np.abs(X).max(axis=1) > threshold


def compute_distance_weights(X: np.ndarray, centers: Any, alpha: float) -> np.ndarray | None:
    """
    Berechnet Gewichte pro Klasse basierend auf Distanz zu Klassen-Zentren im skalierten Raum.
    alpha steuert den Einfluss (höher = stärkere Abwertung bei Distanz).
    """
    if centers is None:
        return None
    centers_arr = np.array(centers, dtype=float)
    if centers_arr.ndim != 2 or centers_arr.shape[1] != X.shape[1]:
        return None
    dists = np.linalg.norm(X[:, None, :] - centers_arr[None, :, :], axis=2)
    weights = np.exp(-alpha * dists)
    sums = weights.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return weights / sums


def interactive_row(feature_cols: List[str]) -> pd.DataFrame:
    print("\nGib Werte zu den Features ein (leer = NA). Erwartet numerische Eingaben.")
    values = {}
    for c in feature_cols:
        val = input(f"{c}: ").strip()
        if val == "":
            values[c] = np.nan
            continue
        if c == "Geschlecht":
            tmp = normalize_gender(pd.DataFrame([{c: val}]))
            values[c] = tmp[c].iloc[0]
            continue
        try:
            values[c] = float(val.replace(",", "."))
        except ValueError:
            values[c] = np.nan
    return pd.DataFrame([values], columns=feature_cols)


def write_template_csv(path: Path, feature_cols: List[str]):
    df = pd.DataFrame(columns=feature_cols)
    df.to_csv(path, index=False)
    print(f"Template geschrieben: {path}")


def load_rf_artifacts(model_path: str) -> Dict[str, Any]:
    data = joblib.load(model_path)
    required = {"model", "preprocessor", "feature_cols", "class_names"}
    if not required.issubset(data.keys()):
        raise ValueError(f"RandomForest-Artefakt unvollständig, erwartet Schlüssel: {sorted(required)}")
    return data


def predict_df(df_new: pd.DataFrame, artifacts: Dict[str, Any], topk: int = 3):
    feature_cols = artifacts["feature_cols"]
    class_names = artifacts["class_names"]
    ood_threshold = artifacts.get("ood_threshold", 0.0)
    dist_mix = artifacts.get("dist_mix", 0.0)
    dist_alpha = artifacts.get("dist_alpha", 1.0)
    X_df = ensure_columns(df_new, feature_cols)
    X_df = normalize_gender(X_df)
    X_df = apply_bounds(X_df, artifacts.get("col_bounds"))
    X = artifacts["preprocessor"].transform(X_df).astype("float32")
    ood_mask = detect_ood(X, ood_threshold)
    dist_weights = compute_distance_weights(X, artifacts.get("class_centers"), dist_alpha) if dist_mix > 0 else None

    model = artifacts["model"]
    probs = model.predict_proba(X)
    if dist_weights is not None:
        probs = (1 - dist_mix) * probs + dist_mix * dist_weights
        probs = probs / probs.sum(axis=1, keepdims=True)

    preds = probs.argmax(axis=1)
    for i, p in enumerate(preds):
        top_idx = probs[i].argsort()[::-1][:topk]
        top = ", ".join([f"{class_names[j]}={probs[i][j]:.3f}" for j in top_idx])
        note = " [OOD]" if ood_mask[i] else ""
        print(f"[{i}] pred: {class_names[p]}  |  top{topk}: {top}{note}")
    return preds, probs


def main():
    ap = argparse.ArgumentParser(description="Inferenz für RandomForest-Baseline")
    ap.add_argument("--csv", type=str, help="CSV mit neuen Fällen.")
    ap.add_argument("--interactive", action="store_true", help="Interaktive Eingabe im Terminal.")
    ap.add_argument("--template", type=str, help="Erzeuge Template-CSV für Eingaben.")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--rf-model", dest="rf_model", type=str, default="outputs/rf/model.joblib")
    ap.add_argument(
        "--ood-threshold",
        type=float,
        default=6.0,
        help="Maximaler |z|-Wert im standardisierten Raum bevor als OOD markiert (<=0 deaktiviert).",
    )
    ap.add_argument(
        "--dist-mix",
        type=float,
        default=0.3,
        help="Mischungsanteil der Distanz-Gewichte (0 = aus).",
    )
    ap.add_argument(
        "--dist-alpha",
        type=float,
        default=1.0,
        help="Steilheit der Distanz-Gewichte (höher = stärkere Abwertung bei Distanz).",
    )
    args = ap.parse_args()

    artifacts = load_rf_artifacts(args.rf_model)
    artifacts["ood_threshold"] = args.ood_threshold
    artifacts["dist_mix"] = args.dist_mix
    artifacts["dist_alpha"] = args.dist_alpha
    feature_cols = artifacts["feature_cols"]

    if args.template:
        write_template_csv(Path(args.template), feature_cols)
        return

    if args.csv:
        df_new = pd.read_csv(args.csv)
    elif args.interactive:
        df_new = interactive_row(feature_cols)
    else:
        print("Bitte --csv PATH oder --interactive angeben (oder --template PATH).")
        return

    _ = predict_df(df_new, artifacts, topk=args.topk)


if __name__ == "__main__":
    main()
