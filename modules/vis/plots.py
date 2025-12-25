from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def _ensure_out_dir(path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def plot_loss_curve(train_loss: List[float], test_loss: List[float], out_path: str, ylabel: str = "Loss"):
    """Zeigt Train/Test-Loss über Epochen. Funktioniert auch mit nur einem Punkt (z.B. SVM/RF)."""
    out = _ensure_out_dir(out_path)
    epochs = list(range(1, len(train_loss) + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_loss, label="Train", marker="o")
    ax.plot(epochs, test_loss, label="Test", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title("Train/Test Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss-Plot gespeichert: {out}")


def plot_pca_decision_regions(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    predict_fn: Callable[[np.ndarray], np.ndarray],
    out_path: str,
    cmap_background: str = "Pastel1",
    cmap_points: str = "tab10",
    random_state: int = 0,
):
    """
    Fit PCA(2) auf X_train (bereits vorverarbeitet/gescaled).
    Zeichnet Decision Regions des übergebenen predict_fn und die Testpunkte in PCA-Raum.
    """
    out = _ensure_out_dir(out_path)
    pca = PCA(n_components=2, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Gitter im PCA-Raum
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid_pca = np.c_[xx.ravel(), yy.ravel()]
    grid_orig = pca.inverse_transform(grid_pca)
    preds_grid = predict_fn(grid_orig).reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    bg_cmap = ListedColormap(plt.get_cmap(cmap_background).colors[: len(class_names)])
    cs = ax.contourf(xx, yy, preds_grid, alpha=0.4, cmap=bg_cmap, levels=np.arange(len(class_names) + 1) - 0.5)
    scatter = ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=cmap_points, edgecolor="k", s=35, alpha=0.9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA + Decision Regions (Test)")
    legend1 = ax.legend(*scatter.legend_elements(), title="Klasse", loc="best", fontsize="small")
    ax.add_artist(legend1)
    fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04, ticks=range(len(class_names)), label="Region")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"PCA/Decision-Plot gespeichert: {out}")


def plot_pca_3d_scatter(
    X_train: np.ndarray,
    X_test: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    out_path: str,
    random_state: int = 0,
):
    """
    Interaktiver PCA(3)-Scatter (HTML), eingefärbt nach labels (z.B. vorhergesagte Klassen).
    Zum Drehen im Browser öffnen.
    """
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot as plotly_plot
    except ImportError as exc:
        raise RuntimeError("Plotly ist nicht installiert (pip install plotly).") from exc

    out = _ensure_out_dir(out_path)
    pca = PCA(n_components=3, random_state=random_state)
    pca.fit(X_train)
    coords = pca.transform(X_test)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    color=labels,
                    colorscale="Turbo",
                    opacity=0.85,
                    line=dict(width=0.5, color="black"),
                ),
                text=[class_names[l] if l < len(class_names) else str(l) for l in labels],
            )
        ]
    )
    fig.update_layout(
        title="PCA 3D (Test)",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    plotly_plot(fig, filename=str(out), auto_open=False, include_plotlyjs="cdn")
    print(f"PCA 3D Plot gespeichert (HTML): {out}")


def plot_pca_3d_correctness(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: str,
    random_state: int = 0,
):
    """
    Interaktiver PCA(3)-Scatter (HTML) mit Grün=korrekt, Rot=falsch.
    Hover zeigt True/Pred-Klasse.
    """
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot as plotly_plot
    except ImportError as exc:
        raise RuntimeError("Plotly ist nicht installiert (pip install plotly).") from exc

    out = _ensure_out_dir(out_path)
    pca = PCA(n_components=3, random_state=random_state)
    pca.fit(X_train)
    coords = pca.transform(X_test)

    correct = (y_true == y_pred)
    traces = []
    for mask, name, color in (
        (correct, "Correct", "green"),
        (~correct, "Wrong", "red"),
    ):
        if not np.any(mask):
            continue
        texts = [
            f"true: {class_names[t] if t < len(class_names) else t}<br>"
            f"pred: {class_names[p] if p < len(class_names) else p}"
            for t, p in zip(y_true[mask], y_pred[mask])
        ]
        traces.append(
            go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode="markers",
                name=name,
                marker=dict(
                    size=4,
                    color=color,
                    opacity=0.9,
                    line=dict(width=0.5, color="black"),
                ),
                text=texts,
                hoverinfo="text",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="PCA 3D – Correct vs Wrong",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    plotly_plot(fig, filename=str(out), auto_open=False, include_plotlyjs="cdn")
    print(f"PCA 3D Correctness Plot gespeichert (HTML): {out}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: str | Path,
    normalize: bool = True,
    cmap: str = "Blues",
):
    """
    Speichert eine Confusion Matrix (optional normiert) als PNG.
    """
    out = _ensure_out_dir(out_path)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        # Zeilenweise normieren; Schutz vor Division durch 0
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, np.clip(row_sums, a_min=1e-12, a_max=None))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Anteil" if normalize else "Anzahl", rotation=-90, va="bottom")

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix" + (" (normiert)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Werte eintragen
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize="small",
            )

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion Matrix gespeichert: {out}")


def plot_nn_history(history: Dict[str, List[float]], out_path: str):
    """Alt: kombinierter Loss/Acc-Plot. Beibehalten für Abwärtskompatibilität."""
    epochs = range(1, len(history["train_loss"]) + 1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["test_loss"], label="Test Loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["test_acc"], label="Test Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Train vs Test Loss/Accuracy")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Trainingskurven gespeichert: {out_path}")


__all__ = [
    "plot_loss_curve",
    "plot_pca_decision_regions",
    "plot_pca_3d_scatter",
    "plot_pca_3d_correctness",
    "plot_confusion_matrix",
    "plot_nn_history",
]
