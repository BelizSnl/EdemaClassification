# LymphDot



# Lymphdoc – Training & Inference

## Struktur
```
LymphdotProject/
├─ modules/
│  ├─ __init__.py
│  ├─ data_prepare.py          # Laden, Split, Label-Encode, Skalierung
│  ├─ utils.py                 # Device/Seed/Dataloader
│  └─ models/
│     ├─ __init__.py
│     └─ mlp.py                # MLPClassifier
├─ train.py                    # Training + Artefakte speichern
├─ inference.py                # Inferenz (CSV oder interaktiv)
├─ outputs/                    # Artefakte (werden erzeugt)
└─ requirements.txt
```

## Training
```bash
conda activate Lymphoma_Classification_Apple
python train.py --data Lymphdoc_medi_4k.csv --target Klassifizierung --epochs 30
```

Artefakte landen in `outputs/`:
- `model.pt` – Weights + Meta (Architektur-HP, Input-Dim, Klassen)
- `preprocessor.joblib` – Skaler + Imputer + Spaltenliste
- `meta.json` – Klassenliste (Fallback)

## Inferenz
- Interaktiv:
  ```bash
  python inference.py --interactive
  ```
- Batch-CSV:
  ```bash
  python inference.py --template new.csv
  # new.csv ausfüllen, dann
  python inference.py --csv new.csv
  ```

## GUI (PyQt5) mit Soft-Voting
- Start: `python scripts/inference/gui.py`
- Wählt per Button eine CSV (gleiche Spalten wie im Training) oder gibt alle Features manuell ein.
- Im Hintergrund holt `scripts/inference/inference_main.py` die Wahrscheinlichkeiten von NN, SVM und RF und mittelt sie per Soft-Voting.
- Die aggregierte Verteilung pro Zeile wird als Popup angezeigt (Top-Klassen mit Wahrscheinlichkeiten).
