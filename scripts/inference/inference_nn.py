#Imports
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.nn.mlp import MLPClassifier

#gerät bestimmen
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

#lade model und preprocessor
def load_artifacts(model_path="outputs/nn/model.pt", preproc_path="outputs/nn/preprocessor.joblib", meta_path="outputs/nn/meta.json"):
    ckpt = torch.load(model_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    n_classes = meta.get("n_classes")
    input_dim  = meta.get("input_dim")
    class_names = meta.get("class_names")
    hparams = meta.get("hparams", {"hidden":[256,128], "p_drop":0.1})

    pre = joblib.load(preproc_path)
    preprocessor = pre["preprocessor"]
    feature_cols: List[str] = pre["feature_cols"]
    col_bounds = pre.get("col_bounds", {})
    class_centers = pre.get("class_centers")

    model = MLPClassifier(input_dim, n_classes, hidden=tuple(hparams.get("hidden",[256,128])), p_drop=hparams.get("p_drop",0.1))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if class_names is None:
        with open(meta_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)["class_names"]

    return model, preprocessor, feature_cols, class_names, col_bounds, class_centers

#sicherstellen, dass alle benötigten spalten vorhanden sind
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


def apply_bounds(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    if not bounds:
        return df
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)
    return df


def compute_distance_weights(X: np.ndarray, centers: dict | list | None, alpha: float) -> np.ndarray | None:
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


def detect_ood(X: np.ndarray, threshold: float) -> np.ndarray:
    """
    Markiert Zeilen als OOD, wenn der maximale |z|-Wert den Schwellwert übersteigt.
    threshold <= 0 deaktiviert die Warnung.
    """
    if threshold is None or threshold <= 0:
        return np.zeros(X.shape[0], dtype=bool)
    return np.abs(X).max(axis=1) > threshold

#vorhersagen für neue daten
@torch.no_grad()
def predict_df(df_new: pd.DataFrame,
               model: nn.Module,
               preprocessor,
               feature_cols: List[str],
               class_names: List[str],
               device: torch.device | None = None,
               topk: int = 3,
               col_bounds: dict | None = None,
               ood_threshold: float = 0.0,
               dist_mix: float = 0.0,
               dist_alpha: float = 1.0,
               class_centers=None):
    device = device or get_device()
    X_df = ensure_columns(df_new, feature_cols)
    X_df = normalize_gender(X_df)
    X_df = apply_bounds(X_df, col_bounds or {})
    X = preprocessor.transform(X_df).astype("float32")
    ood_mask = detect_ood(X, ood_threshold)
    dist_weights = compute_distance_weights(X, class_centers, dist_alpha) if dist_mix > 0 else None
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    model = model.to(device).eval()
    logits = model(xb)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
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

#interaktive eingabe einer zeile
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

#template csv schreiben
def write_template_csv(path: Path, feature_cols: List[str]):
    import pandas as pd
    df = pd.DataFrame(columns=feature_cols)
    df.to_csv(path, index=False)
    print(f"Template geschrieben: {path}")

def main():
    ap = argparse.ArgumentParser(description="Inference for Lymphdot")
    ap.add_argument("--model", default="outputs/nn/model.pt")
    ap.add_argument("--preproc", default="outputs/nn/preprocessor.joblib")
    ap.add_argument("--meta", default="outputs/nn/meta.json")
    ap.add_argument("--csv", type=str, help="CSV mit neuen Fällen (gleiche Roh-Feature-Spalten wie im Training).")
    ap.add_argument("--interactive", action="store_true", help="Interaktiv einen Fall im Terminal eingeben.")
    ap.add_argument("--template", type=str, help="Erzeuge Template-CSV für Eingaben.")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument(
        "--ood-threshold",
        type=float,
        default=5.0,
        help="Maximaler |z|-Wert im standardisierten Raum bevor als OOD markiert (<=0 deaktiviert).",
    )
    ap.add_argument(
        "--dist-mix",
        type=float,
        default=0.2,
        help="Mischungsanteil der Distanz-Gewichte (0 = aus).",
    )
    ap.add_argument(
        "--dist-alpha",
        type=float,
        default=1.0,
        help="Steilheit der Distanz-Gewichte (höher = stärkere Abwertung bei Distanz).",
    )
    args = ap.parse_args()

    model, preprocessor, feature_cols, class_names, col_bounds, class_centers = load_artifacts(args.model, args.preproc, args.meta)

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

    _ = predict_df(
        df_new,
        model,
        preprocessor,
        feature_cols,
        class_names,
        device=get_device(),
        topk=args.topk,
        col_bounds=col_bounds,
        dist_mix=args.dist_mix,
        dist_alpha=args.dist_alpha,
        class_centers=class_centers,
        ood_threshold=args.ood_threshold,
    )

if __name__ == "__main__":
    main()
