# Edema Klassifikation

**Was macht dieses Projekt?**  
Ein Ödem ist eine chronische Schwellung, die durch Flüssigkeitsansammlungen im Gewebe entsteht.  
Dieses Projekt klassifiziert Ödem‑Typen und ‑Stadien aus tabellarischen Patientendaten (CSV). Es trainiert drei Modelle (MLP‑NN, SVM, Random Forest) und bietet eine Inferenz‑Pipeline mit Soft‑Voting‑Ensemble sowie eine GUI für Einzel‑ oder Batch‑Vorhersagen.

Klassen (8):
- gesund
- Lymphödem Stadium 1
- Lymphödem Stadium 2
- Lymphödem Stadium 3
- Lipödem Stadium 1
- Lipödem Stadium 2
- Lipödem Stadium 3
- Lipo-Lymphödem

**Training**  
Einmalig installieren:
```bash
pip install -r requirements.txt
```

Für die beste Inferenzqualität sollten alle drei Modelle (NN, SVM, RF) trainiert werden. Alternativ kann auch nur ein Modell trainiert und dessen Inferenz‑Script genutzt werden.

```bash
python scripts/train/train_nn.py --data Lymphdoc_medi_4k.csv --target Klassifizierung
python scripts/train/train_svm.py --data Lymphdoc_medi_4k.csv --target Klassifizierung
python scripts/train/train_rf.py --data Lymphdoc_medi_4k.csv --target Klassifizierung
```

Outputs & Artefakte werden in `outputs/` gespeichert:
- `outputs/nn/`: NN‑Modell, Preprocessor, Plots
- `outputs/svm/`: SVM‑Modell, Plots
- `outputs/rf/`: RF‑Modell, Plots

**Inferenz**
GUI (empfohlen für schnelle Nutzung):
```bash
python gui/gui.py
```

Ensemble‑Inferenz ohne GUI (NN + SVM + RF, Soft‑Voting):
```bash
python scripts/inference/inference_main.py --csv "PATH/to/file.csv"
```

Einzel‑Modell‑Inferenz (nur CSV):
```bash
python scripts/inference/inference_nn.py --csv "PATH/to/file.csv"
python scripts/inference/inference_svm.py --csv "PATH/to/file.csv"
python scripts/inference/inference_rf.py --csv "PATH/to/file.csv"
```

Interaktiv (Terminal‑Eingabe, nur NN):
```bash
python scripts/inference/inference_nn.py --interactive
python scripts/inference/inference_nn.py --template new.csv
```

**Projektstruktur**
```
.
├─ gui/
│  └─ gui.py
├─ scripts/
│  ├─ analysis/
│  │  ├─ feature_ablation.py
│  │  └─ feature_group_ablation.py
│  ├─ inference/
│  │  ├─ inference_nn.py
│  │  ├─ inference_svm.py
│  │  ├─ inference_rf.py
│  │  └─ inference_main.py
│  └─ train/
│     ├─ train_nn.py
│     ├─ train_svm.py
│     └─ train_rf.py
├─ modules/
│  ├─ nn/        # MLP, Utils
│  ├─ prep/      # Laden, Split, Preprocessing
│  └─ vis/       # Plots
├─ outputs/      # erzeugte Modelle/Plots
│  └─ feature-ablation/
├─ Lymphdoc_medi_4k.csv
└─ requirements.txt
```

**Benötigte Messungen (CSV‑Spalten)**
Für eigene CSV‑Dateien müssen die folgenden Feature‑Spalten verwendet werden.  
Training‑CSVs müssen zusätzlich die Zielspalte `Klassifizierung` enthalten. Inferenz‑CSVs verwenden nur die Feature‑Spalten unten.

```text
Geschlecht (Gender)
Alter (Age)
Größe (Height)
Gewicht (Weight)
Arm links cC (Left arm cC)
Arm links cC1 (Left arm cC1)
Arm links cD (Left arm cD)
Arm links cE (Left arm cE)
Arm links cF (Left arm cF)
Arm links cG (Left arm cG)
Arm rechts cC (Right arm cC)
Arm rechts cC1 (Right arm cC1)
Arm rechts cD (Right arm cD)
Arm rechts cE (Right arm cE)
Arm rechts cF (Right arm cF)
Arm rechts cG (Right arm cG)
Ueber Brust (Above chest)
Unter Brust (Under chest)
Tallie cT (Waist cT)
Hüfte cH (Hip cH)
Bein links cB1 (Left leg cB1)
Bein links cC (Left leg cC)
Bein links cD (Left leg cD)
Bein links cE (Left leg cE)
Bein links cF (Left leg cF)
Bein links cG (Left leg cG)
Bein rechts cB1 (Right leg cB1)
Bein rechts cC (Right leg cC)
Bein rechts cD (Right leg cD)
Bein rechts cE (Right leg cE)
Bein rechts cF (Right leg cF)
Bein rechts cG (Right leg cG)
Druck_links (Pressure left)
Schwere/Trägheit_links (Heaviness/lethargy left)
Taubheit_links (Numbness left)
Schmerz_links (Pain left)
Erwärmung_links (Warmth left)
Druck_rechts (Pressure right)
Schwere/Trägheit_rechts (Heaviness/lethargy right)
Taubheit_rechts (Numbness right)
Schmerz_rechts (Pain right)
Erwärmung_rechts (Warmth right)
```

Mess‑Referenzen:
![Medi arm](gui/Medi_arm.png)
![Medi leg](gui/Medi_bein.png)

**Paper & Plakat**
Paper (PDF im Repo): `paper.pdf`  
Plakat:
![Poster](Plakat.png)
