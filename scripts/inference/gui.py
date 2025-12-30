from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inference.inference_main import EnsembleInference  # type: ignore


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LymphDot Inferenz (Soft-Voting)")
        self.ensemble = EnsembleInference()
        self.topk = 3
        self.inputs: Dict[str, QtWidgets.QLineEdit] = {}
        self._init_ui()

    def _init_ui(self):
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_csv_tab(), "CSV hochladen")
        tabs.addTab(self._build_manual_tab(), "Manuelle Eingabe")
        self.setCentralWidget(tabs)
        self.resize(900, 700)

    def _build_csv_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        info = QtWidgets.QLabel(
            "CSV mit denselben Spalten wie im Training laden. Es können mehrere Zeilen enthalten sein."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        btn = QtWidgets.QPushButton("CSV auswählen und auswerten")
        btn.clicked.connect(self._on_csv_clicked)
        layout.addWidget(btn)
        layout.addStretch()
        return widget

    def _build_manual_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        form_widget = QtWidgets.QWidget()
        form_layout = QtWidgets.QFormLayout(form_widget)

        for col in self.ensemble.feature_cols:
            line = QtWidgets.QLineEdit()
            line.setPlaceholderText("leer lassen für NA")
            self.inputs[col] = line
            form_layout.addRow(col, line)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        layout.addWidget(scroll)

        btn = QtWidgets.QPushButton("Vorhersage berechnen")
        btn.clicked.connect(self._on_manual_clicked)
        layout.addWidget(btn)
        return widget

    def _collect_manual_df(self) -> pd.DataFrame:
        values: Dict[str, float] = {}
        for col, line in self.inputs.items():
            text = line.text().strip()
            if text == "":
                values[col] = np.nan
                continue
            try:
                values[col] = float(text.replace(",", "."))
            except ValueError:
                values[col] = np.nan
        return pd.DataFrame([values], columns=self.ensemble.feature_cols)

    def _show_summary(self, summary: List[str]):
        msg = "\n".join(summary)
        QtWidgets.QMessageBox.information(self, "Vorhersage", msg)

    def _on_csv_clicked(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "CSV auswählen", "", "CSV Dateien (*.csv)")
        if not path:
            return
        try:
            result = self.ensemble.predict_csv(path, topk=self.topk)
            self._show_summary(result["summary"])  # type: ignore[arg-type]
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Fehler", str(exc))

    def _on_manual_clicked(self):
        try:
            df = self._collect_manual_df()
            result = self.ensemble.predict_dataframe(df, topk=self.topk)
            self._show_summary(result["summary"])  # type: ignore[arg-type]
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Fehler", str(exc))


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
