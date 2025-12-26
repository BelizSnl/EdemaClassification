#imports
from __future__ import annotations
import argparse, json, copy
from pathlib import Path
import sys
import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.prep.data_prepare import load_data, split_dataset, fit_label_encoder, encode_labels, scale_features
from modules.prep.feature import ensure_feature
from modules.nn.mlp import MLPClassifier
from modules.nn.utils import set_seed, get_device, device_info, make_dataloaders
from modules.vis.plots import (
    plot_loss_curve,
    plot_pca_decision_regions,
    plot_pca_3d_scatter,
    plot_pca_3d_correctness,
    plot_confusion_matrix,
)

#eine epoche trainiert und validiert
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    #schleife über batches
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)  #Gradienten zurücksetzen
        logits = model(xb)                     #forward pass
        loss = criterion(logits, yb)           #loss berechnen
        loss.backward()                        #backward propagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #Gradienten-Clipping
        optimizer.step()                       #Optimierungsschritt

        running_loss += loss.item() * xb.size(0)  #Gesamtloss
        pred = logits.argmax(dim=1)               #predictions
        correct += (pred == yb).sum().item()      #korrekte Vorhersagen wie viele?
        total   += yb.size(0)                     #insgesamt wie viele?
    return running_loss / total, correct / total

@torch.no_grad() #keine gradientenberechnung für evaluation
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_y, all_p = [], []
    #schleife über batches
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        running_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total   += yb.size(0)
        all_y.append(yb.cpu().numpy()); all_p.append(pred.cpu().numpy())
    y_true = np.concatenate(all_y); y_pred = np.concatenate(all_p) #alle true und pred labels
    return running_loss / total, correct / total, y_true, y_pred

#artefakte speichern
def save_artifacts(preprocessor, class_names, feature_cols, col_bounds, class_centers,
                   preproc_path: str = "outputs/nn/preprocessor.joblib",
                   meta_path: str = "outputs/nn/meta.json"):
    Path(preproc_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "feature_cols": feature_cols,
            "col_bounds": col_bounds,
            "class_centers": class_centers,
        },
        preproc_path,
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names}, f, ensure_ascii=False, indent=2)
    print(f"Artefakte gespeichert: {preproc_path}, {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="Lymphdoc_medi_4k.csv")
    ap.add_argument("--target", type=str, default="Klassifizierung")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, nargs=2, default=[256, 128])
    ap.add_argument("--p_drop", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_path", type=str, default="outputs/nn/model.pt")
    ap.add_argument("--feature", type=str, default="feature.json",
                    help="JSON-Datei mit Feature-Flags (Spaltenname -> bool). Wird automatisch erstellt/aktualisiert.")
    ap.add_argument("--plot-loss", dest="plot_loss", type=str, default="outputs/nn/loss.png")
    ap.add_argument("--plot-pca", dest="plot_pca", type=str, default="outputs/nn/pca_regions.png")
    ap.add_argument("--plot-pca3d", dest="plot_pca3d", type=str, default="outputs/nn/pca_3d.html")
    ap.add_argument(
        "--plot-pca3d-correct",
        dest="plot_pca3d_correct",
        type=str,
        default="outputs/nn/pca_3d_correctness.html",
    )
    ap.add_argument("--plot-cm", dest="plot_cm", type=str, default="outputs/nn/confusion_matrix.png")
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Anzahl Epochen ohne Verbesserung (Test-Loss), bevor abgebrochen wird. 0 = aus.",
    )
    ap.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Mindestverbesserung im Test-Loss, um den Patience-Zähler zurückzusetzen.",
    )
    args = ap.parse_args()

    #gerät und seed setzen
    set_seed(args.seed)
    device = get_device()
    print("Device:", device_info(device))

    #Daten laden
    df = load_data(args.data)

    #Daten vorbereiten
    split = split_dataset(df, target_col=args.target, test_size=0.2, random_state=args.seed) #daten aufteilen
    if args.feature:
        enabled_cols = ensure_feature(args.feature, split.feature_cols)
        split.X_train = split.X_train[enabled_cols]
        split.X_test = split.X_test[enabled_cols]
        split.feature_cols = enabled_cols
    name_to_idx = fit_label_encoder(split.y_train) #label encoder fitten
    enc = encode_labels(split.y_train, split.y_test, name_to_idx) #labels encodieren
    prep = scale_features(split.X_train, split.X_test) #features skalieren

    #Dataloaders erstellen
    train_loader, test_loader = make_dataloaders(prep.X_train, enc.y_train, prep.X_test, enc.y_test, batch_size=args.batch_size, device=device)

    #Modell, Kriterium und Optimierer erstellen
    input_dim = prep.X_train.shape[1]
    n_classes = len(enc.class_names)
    model = MLPClassifier(input_dim, n_classes, hidden=tuple(args.hidden), p_drop=args.p_drop).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #Training Schleife
    best_acc = 0.0
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc, y_true_e, y_pred_e = evaluate(model, test_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true_e, y_pred_e, average="macro", zero_division=0)
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true_e, y_pred_e, average="weighted", zero_division=0)

        if te_acc > best_acc:
            best_acc = te_acc
        if te_loss < best_loss - args.early_stop_min_delta:
            best_loss = te_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        print(f"[{epoch:03d}/{args.epochs}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"test_loss={te_loss:.4f} acc={te_acc:.4f} | "
              f"P_macro={prec_macro:.4f} R_macro={rec_macro:.4f} F1_macro={f1_macro:.4f} | "
              f"P_w={prec_weighted:.4f} R_w={rec_weighted:.4f} F1_w={f1_weighted:.4f}")
        if args.early_stop_patience and no_improve >= args.early_stop_patience:
            print(f"Early Stopping nach {epoch} Epochen (bestes Test-Loss bei Epoche {best_epoch}: {best_loss:.4f})")
            break

    # Plots
    plot_loss_curve(history["train_loss"], history["test_loss"], out_path=args.plot_loss, ylabel="Loss")

    # finale Evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Modell auf bestes Test-Loss zurückgesetzt (Epoche {best_epoch}, loss={best_loss:.4f})")
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print("\n=== TEST REPORT ===")
    print(f"loss={test_loss:.4f}  acc={test_acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=enc.class_names, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred, class_names=enc.class_names, out_path=args.plot_cm)

    # PCA-basierte Plots (Decision Regions + 3D)
    model_cpu = model.to("cpu").eval()

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xb = torch.tensor(arr, dtype=torch.float32)
            logits = model_cpu(xb)
            return logits.softmax(dim=1).argmax(dim=1).cpu().numpy()

    plot_pca_decision_regions(
        X_train=prep.X_train,
        X_test=prep.X_test,
        y_test=y_pred,  # vorhergesagte Labels für Testpunkte (für Farbgebung)
        class_names=enc.class_names,
        predict_fn=predict_fn,
        out_path=args.plot_pca,
        cmap_background="Pastel2",
        cmap_points="tab10",
    )

    plot_pca_3d_scatter(
        X_train=prep.X_train,
        X_test=prep.X_test,
        labels=y_pred,
        class_names=enc.class_names,
        out_path=args.plot_pca3d,
    )
    plot_pca_3d_correctness(
        X_train=prep.X_train,
        X_test=prep.X_test,
        y_true=y_true,
        y_pred=y_pred,
        class_names=enc.class_names,
        out_path=args.plot_pca3d_correct,
    )

    # Speichern
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    meta = {"input_dim": input_dim, "n_classes": n_classes,
            "class_names": enc.class_names,
            "hparams": {"hidden": args.hidden, "p_drop": args.p_drop},
            "best": {"epoch": best_epoch, "val_loss": best_loss, "val_acc": best_acc},
            "early_stopping": {
                "patience": args.early_stop_patience,
                "min_delta": args.early_stop_min_delta,
            }}
    torch.save({"state_dict": model.state_dict(), "meta": meta}, args.save_path)
    print(f"Gespeichert: {args.save_path}")
    # Klassen-Zentren im skalierten Raum für spätere Distanz-Gewichte speichern
    def compute_class_centers(X: np.ndarray, y: np.ndarray, n_classes: int):
        centers = []
        for idx in range(n_classes):
            mask = y == idx
            centers.append(np.mean(X[mask], axis=0).tolist() if np.any(mask) else [float("nan")] * X.shape[1])
        return centers

    class_centers = compute_class_centers(prep.X_train, enc.y_train, n_classes)

    save_artifacts(prep.preprocessor, enc.class_names, split.feature_cols, prep.col_bounds, class_centers)

if __name__ == "__main__":
    main()
