from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtGui, QtCore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inference.inference_main import EnsembleInference  # type: ignore


LOGO_PATH = ROOT / "gui" / "Logo_grau.png"
logo_url = LOGO_PATH.as_posix()

class DropFrame(QtWidgets.QFrame):
    fileDropped = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.setProperty("dragging", False)
        self.overlay: QtWidgets.QWidget | None = None

    def set_overlay(self, overlay: QtWidgets.QWidget) -> None:
        self.overlay = overlay
        self._update_overlay_geometry()

    def _update_overlay_geometry(self):
        if self.overlay:
            self.overlay.setGeometry(self.rect())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_overlay_geometry()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile()
                if path.lower().endswith(".csv"):
                    self.setProperty("dragging", True)
                    self.style().unpolish(self)
                    self.style().polish(self)
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(".csv"):
                self.fileDropped.emit(path)
        self.setProperty("dragging", False)
        self.style().unpolish(self)
        self.style().polish(self)
        event.acceptProposedAction()

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent) -> None:
        self.setProperty("dragging", False)
        self.style().unpolish(self)
        self.style().polish(self)
        event.accept()


class CircularProgress(QtWidgets.QWidget):
    def __init__(self, diameter: int = 110, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._value = 0
        self._diameter = diameter
        self.setFixedSize(diameter, diameter)

    def setValue(self, val: int):
        self._value = max(0, min(100, val))
        self.update()

    def value(self) -> int:
        return self._value

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(6, 6, -6, -6)

        base_pen = QtGui.QPen(QtGui.QColor("#e5e7eb"), 8)
        painter.setPen(base_pen)
        painter.drawArc(rect, 90 * 16, -360 * 16)

        span = int(360 * 16 * (self._value / 100.0))
        progress_pen = QtGui.QPen(QtGui.QColor("#2563eb"), 8)
        painter.setPen(progress_pen)
        painter.drawArc(rect, 90 * 16, -span)

        painter.setPen(QtGui.QPen(QtGui.QColor("#111827")))
        font = QtGui.QFont("Segoe UI", 20)
        painter.setFont(font)
        painter.drawText(self.rect(), QtCore.Qt.AlignCenter, f"{self._value:.0f}%")

STYLESHEET = """
QMainWindow {
    background: transparent;
    color: #111827;
    font-family: 'Segoe UI', 'Helvetica Neue', Arial;
    font-size: 14px;
}

#base {
    background-color: #f5f6fa;
    background-image: url('""" + logo_url + """');
    background-repeat: no-repeat;
    background-position: center;
}

QTabWidget::pane {
    border: 1px solid #d7dce5;
    background: #ffffff;
    border-radius: 10px;
}

QTabBar::tab {
    background: #eef1f6;
    padding: 8px 14px;
    border: 1px solid #d7dce5;
    border-bottom: none;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    color: #111827;
}

QTabBar::tab:selected {
    background: #ffffff;
    border-color: #2563eb;
    font-weight: 600;
    color: #111827;
}

QPushButton {
    background: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 12px 18px;
    font-weight: 600;
}

QPushButton:hover {
    background: #1d4ed8;
}

QPushButton:pressed {
    background: #1e40af;
}

QLineEdit {
    padding: 10px 12px;
    border: 1px solid #2563eb;
    border-radius: 8px;
    background: #ffffff;
    color: #111827;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}

QLineEdit:focus {
    border: 2px solid #1d4ed8;
    outline: none;
}

QLabel {
    color: #111827;
}

QScrollArea {
    border: none;
    background: transparent;
}
"""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LymphDot Inferenz (Soft-Voting)")
        self.ensemble = EnsembleInference()
        self.topk = 3
        self.inputs: Dict[str, QtWidgets.QLineEdit] = {}
        self.drop_card: DropFrame | None = None
        self.drop_overlay: QtWidgets.QFrame | None = None
        self.progress_circle: CircularProgress | None = None
        self.loading_label: QtWidgets.QLabel | None = None
        self.cancel_btn: QtWidgets.QPushButton | None = None
        self.file_icon: QtGui.QPixmap | None = None
        self.loading_timer = QtCore.QTimer(self)
        self.loading_timer.timeout.connect(self._tick_loading)
        self.loading_interval_ms = 50
        self.loading_duration_ms = 2000
        self.loading_step = 100 / (self.loading_duration_ms / self.loading_interval_ms)
        self.pending_path: str | None = None
        self.uploaded_files: list[str] = []
        self.file_list_layout: QtWidgets.QVBoxLayout | None = None
        self.file_items: Dict[str, Dict[str, object]] = {}
        self.start_btn: QtWidgets.QPushButton | None = None
        self._init_ui()

    def _init_ui(self):
        container = QtWidgets.QWidget()
        container.setObjectName("base")
        outer_layout = QtWidgets.QVBoxLayout(container)
        outer_layout.setContentsMargins(20, 20, 20, 20)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_csv_tab(), "CSV hochladen")
        tabs.addTab(self._build_manual_tab(), "Manuelle Eingabe")
        outer_layout.addWidget(tabs)
        self.setCentralWidget(container)
        self.setMinimumSize(1200, 800)

    def _build_csv_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(24)

        # Linke Seite: Drop-Card
        left = QtWidgets.QVBoxLayout()
        left.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)

        drop_card = DropFrame()
        drop_card.setObjectName("dropCard")
        drop_card.setStyleSheet(
            "#dropCard {"
            "  border: 2px dashed #8ab4ff;"
            "  background: #f7f9fc;"
            "  border-radius: 16px;"
            "}"
            "#dropCard[dragging=\"true\"] {"
            "  background: #e1eafe;"
            "}"
            "QLabel { border: none; }"
            "QPushButton { border: none; }"
        )
        drop_card.setAcceptDrops(True)
        drop_card.setFixedSize(420, 420)
        drop_card.fileDropped.connect(self._predict_csv_path)
        self.drop_card = drop_card

        card_layout = QtWidgets.QVBoxLayout(drop_card)
        card_layout.setAlignment(QtCore.Qt.AlignCenter)
        card_layout.setSpacing(10)

        icon_label = QtWidgets.QLabel()
        icon_path = ROOT / "gui" / "Upload.png"
        pixmap = QtGui.QPixmap(str(icon_path)).scaled(120, 120, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        icon_label.setPixmap(pixmap)
        icon_label.setAlignment(QtCore.Qt.AlignCenter)
        card_layout.addWidget(icon_label)

        title = QtWidgets.QLabel("Drop file here")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #111827;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        card_layout.addWidget(title)

        or_label = QtWidgets.QLabel("OR")
        or_label.setStyleSheet("color: #6b7280; font-weight: 600;")
        or_label.setAlignment(QtCore.Qt.AlignCenter)
        card_layout.addWidget(or_label)

        btn = QtWidgets.QPushButton("Upload File")
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setStyleSheet(
            "QPushButton { background: #2563eb; color: #ffffff; border: none; border-radius: 8px; padding: 10px 16px; font-weight: 600; }"
            "QPushButton:hover { background: #1d4ed8; }"
            "QPushButton:pressed { background: #1e40af; }"
        )
        btn.clicked.connect(self._on_csv_clicked)
        card_layout.addWidget(btn)

        hint = QtWidgets.QLabel("Nur CSV-Dateien werden unterstützt.")
        hint.setStyleSheet("color: #6b7280; font-size: 12px;")
        hint.setAlignment(QtCore.Qt.AlignCenter)
        card_layout.addWidget(hint)

        # Overlay für Fake-Upload
        overlay = QtWidgets.QFrame(drop_card)
        overlay.setStyleSheet("QFrame { background: rgba(255,255,255,0.9); border-radius: 16px; }")
        overlay.hide()
        overlay_layout = QtWidgets.QVBoxLayout(overlay)
        overlay_layout.setAlignment(QtCore.Qt.AlignCenter)
        overlay_layout.setSpacing(12)

        progress = CircularProgress()
        overlay_layout.addWidget(progress)

        load_label = QtWidgets.QLabel("Uploading file...")
        load_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #111827;")
        overlay_layout.addWidget(load_label)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setCursor(QtCore.Qt.PointingHandCursor)
        cancel_btn.setStyleSheet(
            "QPushButton { background: #ffffff; color: #2563eb; border: 1px solid #2563eb; border-radius: 8px; padding: 8px 14px; font-weight: 600; }"
            "QPushButton:hover { background: #f0f6ff; }"
            "QPushButton:pressed { background: #e0edff; }"
        )
        cancel_btn.clicked.connect(self._cancel_loading)
        overlay_layout.addWidget(cancel_btn)

        drop_card.set_overlay(overlay)
        self.drop_overlay = overlay
        self.progress_circle = progress
        self.loading_label = load_label
        self.cancel_btn = cancel_btn

        left.addWidget(drop_card)

        # Rechte Seite: Liste + Start
        right = QtWidgets.QVBoxLayout()
        right.setAlignment(QtCore.Qt.AlignTop)
        right.setSpacing(8)
        list_label = QtWidgets.QLabel("Hochgeladene Dateien")
        list_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #111827;")
        right.addWidget(list_label)

        file_icon_path = ROOT / "gui" / "file.png"
        self.file_icon = QtGui.QPixmap(str(file_icon_path)).scaled(20, 20, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        list_container = QtWidgets.QFrame()
        list_container.setStyleSheet("QFrame { border: 1px solid #d7dce5; border-radius: 8px; background: #ffffff; }")
        list_container.setFixedSize(320, 320)
        list_layout = QtWidgets.QVBoxLayout(list_container)
        list_layout.setContentsMargins(8, 8, 8, 8)
        list_layout.setSpacing(8)
        list_layout.addStretch()
        self.file_list_layout = list_layout
        right.addWidget(list_container)

        right.addStretch()
        start_btn = QtWidgets.QPushButton("Start")
        start_btn.setCursor(QtCore.Qt.PointingHandCursor)
        start_btn.setStyleSheet(
            "QPushButton { background: #2563eb; color: #ffffff; border: none; border-radius: 8px; padding: 12px 18px; font-weight: 600; }"
            "QPushButton:hover { background: #1d4ed8; }"
            "QPushButton:pressed { background: #1e40af; }"
        )
        start_btn.clicked.connect(self._on_start_clicked)
        right.addWidget(start_btn)
        self.start_btn = start_btn

        layout.addLayout(left)
        layout.addLayout(right)
        return widget

    def _build_manual_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        content.setObjectName("manualContent")
        content.setStyleSheet("#manualContent { background: #f5f6fa; color: #111827; }")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(18)

        def add_section_title(text: str):
            title = QtWidgets.QLabel(text)
            title.setStyleSheet("font-size: 16px; font-weight: 700; padding: 4px 0;")
            layout.addWidget(title)

        def add_divider():
            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.HLine)
            line.setFrameShadow(QtWidgets.QFrame.Sunken)
            line.setStyleSheet("color: #d7dce5;")
            layout.addWidget(line)

        def add_single_row(label: str, col: str):
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)
            lbl.setMinimumWidth(140)
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText("leer lassen für NA")
            self.inputs[col] = edit
            row.addWidget(lbl)
            row.addWidget(edit)
            layout.addLayout(row)

        def add_pair_row(label: str, col_left: str, col_right: str):
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(12)
            lbl = QtWidgets.QLabel(label)
            lbl.setMinimumWidth(140)
            row.addWidget(lbl)
            # left
            left_box = QtWidgets.QVBoxLayout()
            left_label = QtWidgets.QLabel("Links")
            left_edit = QtWidgets.QLineEdit()
            left_edit.setPlaceholderText("NA")
            self.inputs[col_left] = left_edit
            left_box.addWidget(left_label)
            left_box.addWidget(left_edit)
            # right
            right_box = QtWidgets.QVBoxLayout()
            right_label = QtWidgets.QLabel("Rechts")
            right_edit = QtWidgets.QLineEdit()
            right_edit.setPlaceholderText("NA")
            self.inputs[col_right] = right_edit
            right_box.addWidget(right_label)
            right_box.addWidget(right_edit)

            row.addLayout(left_box)
            row.addLayout(right_box)
            layout.addLayout(row)

        # Überschrift
        header = QtWidgets.QLabel("Messdaten")
        header.setStyleSheet("font-size: 22px; font-weight: 700; padding: 6px 0;")
        layout.addWidget(header)

        # Grundlagen
        add_section_title("Grundlagen")
        add_single_row("Geschlecht", "Geschlecht")
        add_single_row("Alter", "Alter")
        add_single_row("Größe", "Größe")
        add_single_row("Gewicht", "Gewicht")
        add_divider()

        # Messdaten (Maßband)
        add_section_title("Messdaten (Maßband)")
        for base, left_col, right_col in [
            ("Arm cC", "Arm links cC", "Arm rechts cC"),
            ("Arm cC1", "Arm links cC1", "Arm rechts cC1"),
            ("Arm cD", "Arm links cD", "Arm rechts cD"),
            ("Arm cE", "Arm links cE", "Arm rechts cE"),
            ("Arm cF", "Arm links cF", "Arm rechts cF"),
            ("Arm cG", "Arm links cG", "Arm rechts cG"),
            ("Bein cB1", "Bein links cB1", "Bein rechts cB1"),
            ("Bein cC", "Bein links cC", "Bein rechts cC"),
            ("Bein cD", "Bein links cD", "Bein rechts cD"),
            ("Bein cE", "Bein links cE", "Bein rechts cE"),
            ("Bein cF", "Bein links cF", "Bein rechts cF"),
            ("Bein cG", "Bein links cG", "Bein rechts cG"),
        ]:
            add_pair_row(base, left_col, right_col)

        for label, col in [
            ("Über Brust", "Ueber Brust"),
            ("Unter Brust", "Unter Brust"),
            ("Taille cT", "Tallie cT"),
            ("Hüfte cH", "Hüfte cH"),
        ]:
            add_single_row(label, col)
        add_divider()

        # Symptome
        add_section_title("Symptome")
        for base, left_col, right_col in [
            ("Druck", "Druck_links", "Druck_rechts"),
            ("Schwere/Trägheit", "Schwere/Trägheit_links", "Schwere/Trägheit_rechts"),
            ("Taubheit", "Taubheit_links", "Taubheit_rechts"),
            ("Schmerz", "Schmerz_links", "Schmerz_rechts"),
            ("Erwärmung", "Erwärmung_links", "Erwärmung_rechts"),
        ]:
            add_pair_row(base, left_col, right_col)

        # Fallback: noch nicht zugeordnete Features
        handled = set(self.inputs.keys())
        remaining = [c for c in self.ensemble.feature_cols if c not in handled]
        if remaining:
            add_divider()
            add_section_title("Weitere Angaben")
            for col in remaining:
                add_single_row(col, col)

        layout.addStretch()
        btn = QtWidgets.QPushButton("Vorhersage berechnen")
        btn.clicked.connect(self._on_manual_clicked)
        btn.setStyleSheet(
            "QPushButton { background: #2563eb; color: #ffffff; border: none; "
            "border-radius: 10px; padding: 12px 18px; font-weight: 600; } "
            "QPushButton:hover { background: #1d4ed8; } "
            "QPushButton:pressed { background: #1e40af; }"
        )
        layout.addWidget(btn)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setStyleSheet("QScrollArea { border: none; background: #f5f6fa; }")
        scroll.viewport().setStyleSheet("background: #f5f6fa;")
        return scroll

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
        self._predict_csv_path(path)

    def _predict_csv_path(self, path: str):
        if self.loading_timer.isActive():
            return
        self.pending_path = path
        self._ensure_file_item(path)
        self._set_status(path, "Uploading (0%)", "#2563eb")
        if self.progress_circle:
            self.progress_circle.setValue(0)
        if self.drop_overlay:
            self.drop_overlay.show()
        self.loading_timer.start(self.loading_interval_ms)

    def _tick_loading(self):
        if not self.progress_circle:
            return
        val = self.progress_circle.value() + self.loading_step
        if self.pending_path:
            self._set_status(self.pending_path, f"Uploading ({int(min(val,100))}%)", "#2563eb")
        if val >= 100:
            self.progress_circle.setValue(100)
            self.loading_timer.stop()
            QtCore.QTimer.singleShot(100, self._finish_loading)
        else:
            self.progress_circle.setValue(int(val))

    def _finish_loading(self):
        path = self.pending_path
        self.pending_path = None
        if self.drop_overlay:
            self.drop_overlay.hide()
        if path:
            self._add_uploaded_file(path)

    def _cancel_loading(self):
        self.loading_timer.stop()
        path = self.pending_path
        self.pending_path = None
        if self.drop_overlay:
            self.drop_overlay.hide()
        if self.progress_circle:
            self.progress_circle.setValue(0)
        if path:
            self._set_status(path, "Abgebrochen", "#dc2626")
        if path:
            self._set_status(path, "Uploaded", "#16a34a")
            self._add_uploaded_file(path)

    def _add_uploaded_file(self, path: str):
        if path not in self.uploaded_files:
            self.uploaded_files.append(path)

    def _ensure_file_item(self, path: str):
        if path in self.file_items or not self.file_list_layout:
            return
        name = Path(path).name
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(8, 6, 8, 6)
        h.setSpacing(8)
        icon_lbl = QtWidgets.QLabel()
        if self.file_icon:
            icon_lbl.setPixmap(self.file_icon)
        h.addWidget(icon_lbl)
        texts = QtWidgets.QVBoxLayout()
        name_lbl = QtWidgets.QLabel(name)
        name_lbl.setStyleSheet("font-size: 13px; font-weight: 600; color: #111827;")
        status_lbl = QtWidgets.QLabel("Uploading (0%)")
        status_lbl.setStyleSheet("color: #2563eb; font-size: 12px;")
        texts.addWidget(name_lbl)
        texts.addWidget(status_lbl)
        h.addLayout(texts)
        h.addStretch()
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Plain)
        separator.setStyleSheet("color: #e5e7eb;")
        row_layout = QtWidgets.QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addLayout(h)
        row_layout.addWidget(separator)
        if self.file_list_layout.count() > 0:
            self.file_list_layout.insertWidget(self.file_list_layout.count() - 1, row)
        else:
            self.file_list_layout.addWidget(row)
        self.file_items[path] = {"row": row, "status": status_lbl}

    def _set_status(self, path: str, text: str, color: str):
        if path not in self.file_items:
            self._ensure_file_item(path)
        entry = self.file_items.get(path)
        if not entry:
            return
        lbl = entry.get("status")
        if isinstance(lbl, QtWidgets.QLabel):
            lbl.setText(text)
            lbl.setStyleSheet(f"color: {color}; font-size: 12px;")

    def _on_start_clicked(self):
        if not self.uploaded_files:
            QtWidgets.QMessageBox.information(self, "Hinweis", "Bitte zuerst mindestens eine CSV hochladen.")
            return
        summaries: list[str] = []
        for p in self.uploaded_files:
            try:
                result = self.ensemble.predict_csv(p, topk=self.topk)
                summaries.append(f"{Path(p).name}:")
                summaries.extend(result["summary"])  # type: ignore[arg-type]
                summaries.append("")
            except Exception as exc:
                summaries.append(f"{Path(p).name}: Fehler - {exc}")
                summaries.append("")
        if summaries:
            self._show_summary(summaries)

    def _on_manual_clicked(self):
        try:
            df = self._collect_manual_df()
            result = self.ensemble.predict_dataframe(df, topk=self.topk)
            self._show_summary(result["summary"])  # type: ignore[arg-type]
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Fehler", str(exc))


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
