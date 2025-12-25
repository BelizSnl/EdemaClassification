from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from modules.models import MLPClassifier

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_artifacts(model_path="outputs/model.pt", preproc_path="outputs/preprocessor.joblib", meta_path="outputs/meta.json"):
    ckpt = torch.load(model_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    n_classes = meta.get("n_classes")
    input_dim  = meta.get("input_dim")
    class_names = meta.get("class_names")
    hparams = meta.get("hparams", {"hidden":[256,128], "p_drop":0.1})

    pre = joblib.load(preproc_path)
    preprocessor = pre["preprocessor"]
    feature_cols: List[str] = pre["feature_cols"]

    model = MLPClassifier(input_dim, n_classes, hidden=tuple(hparams.get("hidden",[256,128])), p_drop=hparams.get("p_drop",0.1))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if class_names is None:
        with open(meta_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)["class_names"]

    return model, preprocessor, feature_cols, class_names

def ensure_columns(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    return df[feature_cols].copy()

@torch.no_grad()
def predict_df(df_new: pd.DataFrame,
               model: nn.Module,
               preprocessor,
               feature_cols: List[str],
               class_names: List[str],
               device: torch.device | None = None,
               topk: int = 3):
    device = device or get_device()
    X_df = ensure_columns(df_new, feature_cols)
    X = preprocessor.transform(X_df).astype("float32")
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    model = model.to(device).eval()
    logits = model(xb)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    for i, p in enumerate(preds):
        top_idx = probs[i].argsort()[::-1][:topk]
        top = ", ".join([f"{class_names[j]}={probs[i][j]:.3f}" for j in top_idx])
        print(f"[{i}] pred: {class_names[p]}  |  top{topk}: {top}")
    return preds, probs

def interactive_row(feature_cols: List[str]) -> pd.DataFrame:
    print("\nGib Werte zu den Features ein (leer = NA). Erwartet numerische Eingaben.")
    values = {}
    for c in feature_cols:
        val = input(f"{c}: ").strip()
        values[c] = (float(val.replace(",", ".")) if val != "" else np.nan)
    return pd.DataFrame([values], columns=feature_cols)

def write_template_csv(path: Path, feature_cols: List[str]):
    import pandas as pd
    df = pd.DataFrame(columns=feature_cols)
    df.to_csv(path, index=False)
    print(f"Template geschrieben: {path}")

def main():
    ap = argparse.ArgumentParser(description="Inference for Lymphdot")
    ap.add_argument("--model", default="outputs/model.pt")
    ap.add_argument("--preproc", default="outputs/preprocessor.joblib")
    ap.add_argument("--meta", default="outputs/meta.json")
    ap.add_argument("--csv", type=str, help="CSV mit neuen Fällen (gleiche Roh-Feature-Spalten wie im Training).")
    ap.add_argument("--interactive", action="store_true", help="Interaktiv einen Fall im Terminal eingeben.")
    ap.add_argument("--template", type=str, help="Erzeuge Template-CSV für Eingaben.")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    model, preprocessor, feature_cols, class_names = load_artifacts(args.model, args.preproc, args.meta)

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

    _ = predict_df(df_new, model, preprocessor, feature_cols, class_names, device=get_device(), topk=args.topk)

if __name__ == "__main__":
    main()
