#imports
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ---------------- Data Classes -----------------
@dataclass
class SplitResult: 
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_cols: List[str]

@dataclass
class LabelEncoderResult: 
    name_to_idx: Dict[str, int]
    class_names: List[str]
    y_train: np.ndarray
    y_test: np.ndarray

@dataclass
class PreprocessResult: 
    X_train: np.ndarray
    X_test: np.ndarray
    preprocessor: ColumnTransformer
    cont_cols: List[str]
    bin_cols: List[str]
    col_bounds: Dict[str, Tuple[float, float]]

# ---------------- Dataset -----------------
def load_data(data_path):
    data=pd.read_csv(data_path)
    print(f"Data load Done hehe")
    return data

# ---------------- train test split -----------------
def split_dataset(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitResult:
    if target_col not in df.columns:
        raise KeyError(f"Zielspalte '{target_col}' nicht gefunden.")
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return SplitResult(X_train, X_test, y_train, y_test, feature_cols)

# ---------------- label-encoding ----------------
def fit_label_encoder(y_train: pd.Series) -> Dict[str, int]:
    class_names = sorted(y_train.unique().tolist())
    return {name: i for i, name in enumerate(class_names)}

def encode_labels(y_train: pd.Series, y_test: pd.Series, name_to_idx: Dict[str, int]) -> LabelEncoderResult:
    unknown_test = set(y_test.unique()) - set(name_to_idx.keys())
    if unknown_test:
        raise ValueError(f"Unbekannte Klassen im Test-Set: {sorted(unknown_test)}")
    class_names = [k for k,_ in sorted(name_to_idx.items(), key=lambda kv: kv[1])]
    ytr = y_train.map(name_to_idx).astype("int64").values
    yte = y_test.map(name_to_idx).astype("int64").values
    return LabelEncoderResult(name_to_idx=name_to_idx, class_names=class_names, y_train=ytr, y_test=yte)

# ---------- Skalierung ----------
def scale_features(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame) -> PreprocessResult:
    # Kopien anlegen, damit wir Spalten gefahrlos transformieren können
    X_train_df = X_train_df.copy()
    X_test_df = X_test_df.copy()

    # Geschlecht als numerisch kodieren (männlich=0, weiblich=1)
    if "Geschlecht" in X_train_df.columns:
        gender_map = {"männlich": 0, "weiblich": 1}

        def _map_gender(series: pd.Series) -> pd.Series:
            return series.apply(
                lambda v: gender_map.get(str(v).strip().lower()) if pd.notna(v) else np.nan
            )

        X_train_df["Geschlecht"] = _map_gender(X_train_df["Geschlecht"])
        X_test_df["Geschlecht"] = _map_gender(X_test_df["Geschlecht"])

    num_cols = X_train_df.select_dtypes(include=["number", "bool"]).columns.tolist()

    # Per-Feature-Grenzen aus Trainingsdaten ableiten (robust über Quantile)
    col_bounds: Dict[str, Tuple[float, float]] = {}
    for col in num_cols:
        s = X_train_df[col].dropna()
        if s.empty:
            continue
        lower = float(s.quantile(0.01))
        upper = float(s.quantile(0.99))
        # Fallbacks bei degenerierten Quantilen
        if not np.isfinite(lower):
            lower = float(s.min())
        if not np.isfinite(upper):
            upper = float(s.max())
        if lower > upper:
            lower, upper = upper, lower
        col_bounds[col] = (lower, upper)
        X_train_df[col] = X_train_df[col].clip(lower, upper)
        if col in X_test_df:
            X_test_df[col] = X_test_df[col].clip(lower, upper)

    bin_cols = [c for c in num_cols if X_train_df[c].dropna().isin([0, 1]).all() and X_train_df[c].nunique(dropna=True) <= 2]
    cont_cols = [c for c in num_cols if c not in bin_cols]

    transformers = []
    if cont_cols:
        transformers.append(("cont", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), cont_cols))
    if bin_cols:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), bin_cols))
    if not transformers:
        raise ValueError("Keine geeigneten numerischen Features gefunden.")

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    X_train = pre.fit_transform(X_train_df).astype("float32")
    X_test  = pre.transform(X_test_df).astype("float32")
    return PreprocessResult(
        X_train=X_train,
        X_test=X_test,
        preprocessor=pre,
        cont_cols=cont_cols,
        bin_cols=bin_cols,
        col_bounds=col_bounds,
    )
