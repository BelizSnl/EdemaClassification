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


def interactive_row(feature_cols: List[str]) -> pd.DataFrame:
    print("\nGib Werte zu den Features ein (leer = NA). Erwartet numerische Eingaben.")
    values = {}
    for c in feature_cols:
        val = input(f"{c}: ").strip()
        values[c] = (float(val.replace(",", ".")) if val != "" else np.nan)
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
    X_df = ensure_columns(df_new, feature_cols)
    X = artifacts["preprocessor"].transform(X_df).astype("float32")

    model = artifacts["model"]
    probs = model.predict_proba(X)

    preds = probs.argmax(axis=1)
    for i, p in enumerate(preds):
        top_idx = probs[i].argsort()[::-1][:topk]
        top = ", ".join([f"{class_names[j]}={probs[i][j]:.3f}" for j in top_idx])
        print(f"[{i}] pred: {class_names[p]}  |  top{topk}: {top}")
    return preds, probs


def main():
    ap = argparse.ArgumentParser(description="Inferenz für RandomForest-Baseline")
    ap.add_argument("--csv", type=str, help="CSV mit neuen Fällen.")
    ap.add_argument("--interactive", action="store_true", help="Interaktive Eingabe im Terminal.")
    ap.add_argument("--template", type=str, help="Erzeuge Template-CSV für Eingaben.")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--rf-model", dest="rf_model", type=str, default="outputs/rf/model.joblib")
    args = ap.parse_args()

    artifacts = load_rf_artifacts(args.rf_model)
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
