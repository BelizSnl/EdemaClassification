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
python train.py --data Lymphdoc_medi_gesammtdaten.csv --target Klassifizierung --epochs 30
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
