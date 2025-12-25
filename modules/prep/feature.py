from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict


def _normalize_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return True


def ensure_feature(config_path: str | Path, feature_cols: Iterable[str]) -> List[str]:
    """
    Ensure a toggle file exists and return the columns that are enabled (True).
    Creates the file if missing and adds any newly seen columns automatically.
    """
    feature_cols = list(feature_cols)
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    toggles: Dict[str, bool] = {}
    file_existed = path.exists()
    if file_existed:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Feature-Flag-Datei '{path}' ist kein gültiges JSON ({exc}).") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Feature-Flag-Datei '{path}' muss ein Objekt aus Spaltennamen -> bool sein.")
        toggles = {str(k): _normalize_flag(v) for k, v in data.items()}

    selected_cols: List[str] = []
    needs_update = not file_existed
    ordered: Dict[str, bool] = {}

    for col in feature_cols:
        enabled = toggles.get(col, True)
        if col not in toggles:
            needs_update = True
        ordered[col] = bool(enabled)
        if enabled:
            selected_cols.append(col)

    extra_toggles = [c for c in toggles.keys() if c not in ordered]
    if extra_toggles:
        # Keep extras (maybe columns no longer present) at the end to avoid silent loss.
        for col in extra_toggles:
            ordered[col] = toggles[col]
        needs_update = True

    if needs_update:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(ordered, fh, ensure_ascii=False, indent=2)
        if file_existed:
            print(f"Feature-Flag-Datei aktualisiert: {path}")
        else:
            print(f"Feature-Flag-Datei erstellt: {path}")

    if extra_toggles:
        print(f"Warnung: {len(extra_toggles)} Spalten im Toggle-File nicht in den Daten gefunden: {extra_toggles}")

    if not selected_cols:
        raise ValueError(f"In '{path}' sind alle Spalten deaktiviert – es muss mindestens eine aktiv sein.")

    return selected_cols


__all__ = ["ensure_feature"]