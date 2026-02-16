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
    """Frame that accepts CSV drag-and-drop and emits the selected path."""
    fileDropped = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        """Initialize drag-and-drop state."""
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.setProperty("dragging", False)
        self.overlay: QtWidgets.QWidget | None = None

    def set_overlay(self, overlay: QtWidgets.QWidget) -> None:
        """Attach a visual overlay and keep it aligned with the frame."""
        self.overlay = overlay
        self._update_overlay_geometry()

    def _update_overlay_geometry(self):
        """Keep the overlay size in sync with the frame size."""
        if self.overlay:
            self.overlay.setGeometry(self.rect())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Update overlay when the frame is resized."""
        super().resizeEvent(event)
        self._update_overlay_geometry()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept CSV drags and update the visual state."""
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
        """Emit the dropped CSV path and reset the visual state."""
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
        """Reset visual state when drag leaves the frame."""
        self.setProperty("dragging", False)
        self.style().unpolish(self)
        self.style().polish(self)
        event.accept()


class CircularProgress(QtWidgets.QWidget):
    """Circular progress indicator for long-running operations."""
    def __init__(self, diameter: int = 110, parent: QtWidgets.QWidget | None = None):
        """Create a circular progress widget with a fixed diameter."""
        super().__init__(parent)
        self._value = 0
        self._diameter = diameter
        self.setFixedSize(diameter, diameter)

    def setValue(self, val: int):
        """Set progress value (0–100) and repaint."""
        self._value = max(0, min(100, val))
        self.update()

    def value(self) -> int:
        """Return current progress value."""
        return self._value

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Draw the progress arc and numeric percentage."""
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

QScrollBar {
    width: 10px;
    background: #e5e7eb;
    border-radius: 5px;
}

QScrollBar::handle {
    background: #cbd5e1;
    border-radius: 5px;
    min-height: 24px;
}

QScrollBar::add-line, QScrollBar::sub-line {
    background: transparent;
    height: 0px;
}

QScrollBar::add-page, QScrollBar::sub-page {
    background: #e5e7eb;
    border-radius: 5px;
}
"""


class MainWindow(QtWidgets.QMainWindow):
    """Main application window for CSV upload and manual input inference."""
    def __init__(self):
        """Initialize state, inference backend, and UI."""
        super().__init__()
        self.setWindowTitle("Benutzeroberfläche")
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
        self.file_list_container: QtWidgets.QFrame | None = None
        self.list_label_widget: QtWidgets.QLabel | None = None
        self.file_items: Dict[str, Dict[str, object]] = {}
        self.start_btn: QtWidgets.QPushButton | None = None
        self._init_ui()

    def _init_ui(self):
        """Build the main layout, top bar, and content container."""
        container = QtWidgets.QWidget()
        container.setObjectName("base")
        outer_layout = QtWidgets.QVBoxLayout(container)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        # Top bar
        top_bar = QtWidgets.QFrame()
        bg_path = (ROOT / "gui" / "hintergrund_leiste_cropped.png").as_posix()
        top_bar.setStyleSheet(
            f"QFrame {{ background-color: #0d4e6b; background-image: url('{bg_path}'); "
            f"background-repeat: no-repeat; background-position: center center; background-size: 100% 140px; }}"
        )
        top_bar.setFixedHeight(140)
        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QtGui.QColor(0, 0, 0, 40))
        top_bar.setGraphicsEffect(shadow)
        bar_layout = QtWidgets.QHBoxLayout(top_bar)
        bar_layout.setContentsMargins(24, 20, 24, 20)

        bar_layout.addStretch(2)
        title = QtWidgets.QLabel(
            "Stadieneinteilung von Lipödem, Lymphödem und Lipo-Lymphödem"
        )
        title.setStyleSheet("font-size: 30px; font-weight: 400; color: #ffffff;")
        bar_layout.addWidget(title, alignment=QtCore.Qt.AlignCenter)
        bar_layout.addStretch(1)

        hsd_logo_path = ROOT / "gui" / "HSDLOGO.png"
        if hsd_logo_path.exists():
            logo_lbl = QtWidgets.QLabel()
            pix = QtGui.QPixmap(str(hsd_logo_path)).scaled(
                192, 192, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            logo_lbl.setPixmap(pix)
            logo_lbl.setStyleSheet("background: transparent;")
            logo_lbl.setAttribute(QtCore.Qt.WA_TranslucentBackground)
            logo_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            bar_layout.addWidget(logo_lbl)

        outer_layout.addWidget(top_bar)

        content_wrapper = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content_wrapper)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.addWidget(self._build_manual_tab())
        outer_layout.addWidget(content_wrapper)
        self.setCentralWidget(container)
        self.setMinimumSize(1200, 800)

    def _build_upload_section(self) -> QtWidgets.QWidget:
        """Create the CSV upload panel with drop area and file list."""
        frame = QtWidgets.QFrame()
        frame.setObjectName("uploadFrame")
        frame.setStyleSheet(
            "#uploadFrame { border: 1px solid #d7dce5; border-radius: 8px; background: transparent; }"
        )
        main_layout = QtWidgets.QVBoxLayout(frame)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)
        title_upload = QtWidgets.QLabel("CSV Eingabe")
        title_upload.setStyleSheet(
            "font-size: 22px; font-weight: 700; padding: 6px 0; color: #111827;"
        )
        main_layout.addWidget(title_upload, alignment=QtCore.Qt.AlignLeft)

        layout = QtWidgets.QHBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        layout.setSpacing(8)

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
            '#dropCard[dragging="true"] {'
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
        pixmap = QtGui.QPixmap(str(icon_path)).scaled(
            120, 120, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
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
        overlay.setStyleSheet(
            "QFrame { background: rgba(255,255,255,0.9); border-radius: 16px; }"
        )
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
        self.list_label_widget = list_label

        file_icon_path = ROOT / "gui" / "file.png"
        self.file_icon = QtGui.QPixmap(str(file_icon_path)).scaled(
            20, 20, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

        list_container = QtWidgets.QFrame()
        list_container.setStyleSheet(
            "QFrame { border: 1px solid #f5f9fe; border-radius: 8px; background: transparent; }"
        )
        list_container.setFixedSize(420, 320)
        list_layout = QtWidgets.QVBoxLayout(list_container)
        list_layout.setContentsMargins(8, 8, 8, 8)
        list_layout.setSpacing(8)
        list_layout.addStretch()
        self.file_list_layout = list_layout
        self.file_list_container = list_container
        right.addWidget(list_container)

        right.addStretch()
        start_btn = QtWidgets.QPushButton("Vorhersage berechnen")
        start_btn.setCursor(QtCore.Qt.PointingHandCursor)
        start_btn.setFixedWidth(200)
        start_btn.setStyleSheet(
            "QPushButton { background: #2563eb; color: #ffffff; border: none; border-radius: 8px; padding: 12px 14px; font-weight: 600; }"
            "QPushButton:hover { background: #1d4ed8; }"
            "QPushButton:pressed { background: #1e40af; }"
        )
        start_btn.clicked.connect(self._on_start_clicked)
        right.addWidget(start_btn)
        self.start_btn = start_btn

        # Initial hiding until first upload
        if self.list_label_widget:
            self.list_label_widget.hide()
        if self.file_list_container:
            self.file_list_container.hide()
        if self.start_btn:
            self.start_btn.hide()

        layout.addLayout(left)
        layout.addLayout(right)
        main_layout.addLayout(layout)
        return frame

    def _build_manual_tab(self) -> QtWidgets.QWidget:
        """Create the manual input form with grouped measurement fields."""
        content = QtWidgets.QWidget()
        content.setObjectName("manualContent")
        content.setStyleSheet("#manualContent { background: #f5f6fa; color: #111827; }")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(22)

        upload_section = self._build_upload_section()
        layout.addWidget(upload_section)
        or_container = QtWidgets.QWidget()
        or_layout = QtWidgets.QHBoxLayout(or_container)
        or_layout.setContentsMargins(0, 8, 0, 12)
        or_layout.setSpacing(12)
        line_left = QtWidgets.QFrame()
        line_left.setFixedHeight(1)
        line_left.setStyleSheet("QFrame { background-color: #d7dce5; border: none; }")
        line_right = QtWidgets.QFrame()
        line_right.setFixedHeight(1)
        line_right.setStyleSheet("QFrame { background-color: #d7dce5; border: none; }")
        or_label = QtWidgets.QLabel("ODER")
        or_label.setStyleSheet(
            "font-size: 22px; font-weight: 700; color: #111827; padding: 6px 0;"
        )
        or_layout.addWidget(line_left, 1)
        or_layout.addWidget(or_label)
        or_layout.addWidget(line_right, 1)
        layout.addWidget(or_container)

        form_frame = QtWidgets.QFrame()
        form_frame.setObjectName("formFrame")
        form_frame.setStyleSheet(
            "#formFrame { border: 1px solid #d7dce5; border-radius: 8px; background: transparent; }"
        )
        form_layout = QtWidgets.QVBoxLayout(form_frame)
        form_layout.setContentsMargins(12, 12, 12, 12)
        form_layout.setSpacing(18)

        def add_section_title(text: str):
            """Insert a section title label into the form layout."""
            title = QtWidgets.QLabel(text)
            title.setStyleSheet("font-size: 16px; font-weight: 700; padding: 4px 0;")
            form_layout.addWidget(title)

        def add_divider():
            """Insert a horizontal divider line."""
            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.HLine)
            line.setFrameShadow(QtWidgets.QFrame.Sunken)
            line.setStyleSheet("color: #d7dce5;")
            form_layout.addWidget(line)

        def add_single_row(label: str, col: str):
            """Add a single input row and register its line edit."""
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)
            lbl.setMinimumWidth(140)
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText("leer lassen für NA")
            self.inputs[col] = edit
            row.addWidget(lbl)
            row.addWidget(edit)
            form_layout.addLayout(row)

        def add_pair_row(label: str, col_left: str, col_right: str):
            """Add a paired left/right input row and register both fields."""
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
            form_layout.addLayout(row)

        # Überschrift
        header = QtWidgets.QLabel("Messdaten")
        header.setStyleSheet("font-size: 22px; font-weight: 700; padding: 6px 0;")
        form_layout.addWidget(header)

        # Grundlagen
        add_section_title("Grundlagen")
        add_single_row("Geschlecht", "Geschlecht")
        add_single_row("Alter", "Alter")
        add_single_row("Größe", "Größe")
        add_single_row("Gewicht", "Gewicht")
        add_divider()

        # Messdaten (Maßband)
        add_section_title("Messdaten (Maßband)")
        # Build paired left/right measurement rows.
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

        # Add single-body measurement fields.
        for label, col in [
            ("Über Brust", "Ueber Brust"),
            ("Unter Brust", "Unter Brust"),
            ("Taille cT", "Tallie cT"),
            ("Hüfte cH", "Hüfte cH"),
        ]:
            add_single_row(label, col)

        # Referenzgrafiken für Umfangsmessung
        illustration_row = QtWidgets.QHBoxLayout()
        illustration_row.setSpacing(16)
        illustration_row.setAlignment(QtCore.Qt.AlignCenter)
        has_illustration = False
        # Add measurement reference images if available.
        for img_name in ["Medi_arm.png", "Medi_bein.png"]:
            img_path = ROOT / "gui" / img_name
            if not img_path.exists():
                continue
            pix = QtGui.QPixmap(str(img_path))
            if pix.isNull():
                continue
            lbl = QtWidgets.QLabel()
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setPixmap(
                pix.scaled(
                    320, 260, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                )
            )
            lbl.setStyleSheet(
                "QLabel {"
                " background: #ffffff;"
                " border: 1px solid #2563eb;"
                " border-radius: 8px;"
                " padding: 8px;"
                "}"
            )
            illustration_row.addWidget(lbl)
            has_illustration = True
        if has_illustration:
            form_layout.addLayout(illustration_row)
        add_divider()

        # Symptome
        add_section_title("Symptome")
        # Build paired symptom rows.
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
            # Add any remaining features not mapped above.
            for col in remaining:
                add_single_row(col, col)

        layout.addWidget(form_frame)

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
        """Collect manual input fields into a single-row DataFrame."""
        values: Dict[str, float] = {}
        # Read all input fields and coerce to float/NaN.
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

    def _show_message(
        self,
        title: str,
        text: str,
        icon: QtWidgets.QMessageBox.Icon = QtWidgets.QMessageBox.Information,
    ) -> None:
        """Show a styled message box to the user."""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        box.setIcon(icon)
        box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        box.setStyleSheet(
            "QMessageBox { background-color: #ffffff; font-family: 'Segoe UI', 'Helvetica Neue', Arial; font-size: 14px; color: #111827; }"
            "QLabel { color: #111827; }"
            "QPushButton { background: #2563eb; color: #ffffff; border: none; border-radius: 8px; padding: 8px 14px; font-weight: 600; min-width: 80px; }"
            "QPushButton:hover { background: #1d4ed8; }"
            "QPushButton:pressed { background: #1e40af; }"
        )
        box.exec_()

    def _prepare_entries(
        self, result: Dict[str, object], header: str | None = None
    ) -> list[tuple[str, list[tuple[str, str]]]]:
        """Format model output into display-ready entries."""
        class_names = result.get("class_names")
        probs = result.get("avg_probs")
        preds = result.get("preds")
        if class_names is None or probs is None or preds is None:
            fallback = result.get("summary", [])  # type: ignore[arg-type]
            text_lines = fallback if isinstance(fallback, list) else [str(fallback)]
            return [(header or "Vorhersage", [(line, "") for line in text_lines])]

        probs_arr = np.asarray(probs)
        preds_arr = np.asarray(preds)
        entries: list[tuple[str, list[tuple[str, str]]]] = []
        # Convert per-row probabilities to top-k display lines.
        for i in range(len(preds_arr)):
            lines: list[tuple[str, str]] = []
            order = probs_arr[i].argsort()[::-1][: self.topk]
            # Add the top-k classes with their percentages.
            for idx in order:
                lines.append((class_names[int(idx)], f"{probs_arr[i][idx] * 100:.1f}%"))
            title = header or "Vorhersage"
            if len(probs_arr) > 1:
                title = f"{title} — Fall {i + 1}"
            entries.append((title, lines))
        return entries

    def _build_result_card(
        self, title: str, rows: list[tuple[str, str]]
    ) -> QtWidgets.QFrame:
        """Create a styled card widget for one prediction result."""
        card = QtWidgets.QFrame()
        card.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; }"
        )
        card.setMinimumWidth(420)
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(12)

        # Header row: badge + title
        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(8)
        badge = QtWidgets.QLabel("PREDICTION")
        badge.setStyleSheet(
            "background: #e0edff; color: #2563eb; font-size: 11px; font-weight: 700; "
            "padding: 4px 8px; border-radius: 8px; letter-spacing: 0.5px;"
        )
        header_row.addWidget(badge, 0, QtCore.Qt.AlignLeft)
        header_row.addStretch()
        lay.addLayout(header_row)

        header = QtWidgets.QLabel(title)
        header.setFrameStyle(QtWidgets.QFrame.NoFrame)
        header.setStyleSheet(
            "font-size: 17px; font-weight: 700; color: #111827; border: none; background: transparent;"
        )
        lay.addWidget(header)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.NoFrame)
        line.setFrameShadow(QtWidgets.QFrame.Plain)
        line.setFixedHeight(2)
        line.setStyleSheet("background-color: #e5e7eb; border: none;")
        lay.addWidget(line)

        # Render each prediction row.
        for label, value in rows:
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(8)
            lbl = QtWidgets.QLabel(label)
            lbl.setFrameStyle(QtWidgets.QFrame.NoFrame)
            lbl.setStyleSheet(
                "color: #6b7280; font-size: 13px; background: transparent; border: none; padding: 0;"
            )
            val = QtWidgets.QLabel(value)
            val.setFrameStyle(QtWidgets.QFrame.NoFrame)
            val.setStyleSheet(
                "color: #111827; font-weight: 700; font-size: 15px; background: transparent; border: none; padding: 0;"
            )
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val, 0, QtCore.Qt.AlignRight)
            lay.addLayout(row)

        return card

    def _show_result_dialog(
        self, entries: list[tuple[str, list[tuple[str, str]]]]
    ) -> None:
        """Show the prediction results in a modal dialog."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Vorhersage")
        dialog.setModal(True)
        dialog.setStyleSheet(
            "QDialog { background: #f1f5f9; font-family: 'Segoe UI', 'Helvetica Neue', Arial; font-size: 14px; }"
            "QScrollArea { border: none; background: transparent; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
            "QScrollBar { width: 10px; background: #e5e7eb; border-radius: 5px; }"
            "QScrollBar::handle { background: #cbd5e1; border-radius: 5px; min-height: 24px; }"
            "QScrollBar::add-line, QScrollBar::sub-line { background: transparent; height: 0px; }"
            "QScrollBar::add-page, QScrollBar::sub-page { background: #e5e7eb; border-radius: 5px; }"
            "QPushButton { background: #2563eb; color: #ffffff; border: none; border-radius: 8px; padding: 10px 16px; font-weight: 600; }"
            "QPushButton:hover { background: #1d4ed8; }"
            "QPushButton:pressed { background: #1e40af; }"
        )

        outer_layout = QtWidgets.QVBoxLayout(dialog)
        outer_layout.setContentsMargins(20, 20, 20, 20)
        outer_layout.setSpacing(14)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        wrapper = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(wrapper)
        vbox.setContentsMargins(0, 0, 10, 0)
        vbox.setSpacing(12)

        # Build a card for each entry.
        for title, rows in entries:
            vbox.addWidget(self._build_result_card(title, rows))

        scroll.setWidget(wrapper)
        outer_layout.addWidget(scroll)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        btn_row.addWidget(ok_btn)
        outer_layout.addLayout(btn_row)

        dialog.resize(520, 480)
        dialog.exec_()

    def _on_csv_clicked(self):
        """Open file dialog and start CSV inference."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "CSV auswählen", "", "CSV Dateien (*.csv)"
        )
        if not path:
            return
        self._predict_csv_path(path)

    def _predict_csv_path(self, path: str):
        """Start the simulated upload flow for a selected CSV."""
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
        """Advance the upload progress animation."""
        if not self.progress_circle:
            return
        val = self.progress_circle.value() + self.loading_step
        if self.pending_path:
            self._set_status(
                self.pending_path, f"Uploading ({int(min(val,100))}%)", "#2563eb"
            )
        if val >= 100:
            self.progress_circle.setValue(100)
            self.loading_timer.stop()
            QtCore.QTimer.singleShot(100, self._finish_loading)
        else:
            self.progress_circle.setValue(int(val))

    def _finish_loading(self):
        """Finalize upload and add file to the list."""
        path = self.pending_path
        self.pending_path = None
        if self.drop_overlay:
            self.drop_overlay.hide()
        if path:
            self._add_uploaded_file(path)

    def _cancel_loading(self):
        """Cancel upload and reset UI state."""
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
        """Add a file to the internal list and reveal widgets."""
        if path not in self.uploaded_files:
            self.uploaded_files.append(path)
            if len(self.uploaded_files) == 1:
                self._show_upload_widgets()

    def _remove_uploaded_file(self, path: str):
        """Remove a file entry from the list and hide widgets if empty."""
        if path in self.uploaded_files:
            self.uploaded_files.remove(path)
        entry = self.file_items.pop(path, None)
        if entry and "row" in entry and isinstance(entry["row"], QtWidgets.QWidget):
            row_widget: QtWidgets.QWidget = entry["row"]  # type: ignore[assignment]
            row_widget.setParent(None)
            row_widget.deleteLater()
        if not self.uploaded_files:
            if self.list_label_widget:
                self.list_label_widget.hide()
            if self.file_list_container:
                self.file_list_container.hide()
            if self.start_btn:
                self.start_btn.hide()

    def _ensure_file_item(self, path: str):
        """Create a file row widget for the upload list if missing."""
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
        name_lbl.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        status_lbl = QtWidgets.QLabel("Uploading (0%)")
        status_lbl.setStyleSheet("color: #2563eb; font-size: 12px;")
        texts.addWidget(name_lbl)
        texts.addWidget(status_lbl)
        h.addLayout(texts, 1)
        h.addStretch()

        delete_btn = QtWidgets.QPushButton("Löschen")
        delete_btn.setCursor(QtCore.Qt.PointingHandCursor)
        delete_btn.setStyleSheet(
            "QPushButton { background: #ffffff; color: #dc2626; border: 1px solid #dc2626; border-radius: 6px; padding: 6px 10px; font-weight: 600; }"
            "QPushButton:hover { background: #fff5f5; }"
            "QPushButton:pressed { background: #ffe4e6; }"
        )
        delete_btn.clicked.connect(lambda _, p=path: self._remove_uploaded_file(p))
        h.addWidget(delete_btn)

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
        """Update status text and color for an uploaded file."""
        if path not in self.file_items:
            self._ensure_file_item(path)
        entry = self.file_items.get(path)
        if not entry:
            return
        lbl = entry.get("status")
        if isinstance(lbl, QtWidgets.QLabel):
            lbl.setText(text)
            lbl.setStyleSheet(f"color: {color}; font-size: 12px;")

    def _show_upload_widgets(self):
        """Reveal upload list and action buttons once files exist."""
        if self.list_label_widget:
            self.list_label_widget.show()
        if self.file_list_container:
            self.file_list_container.show()
        if self.start_btn:
            self.start_btn.show()

    def _on_start_clicked(self):
        """Run ensemble inference for all queued CSV files."""
        if not self.uploaded_files:
            self._show_message(
                "Hinweis",
                "Bitte zuerst mindestens eine CSV hochladen.",
                QtWidgets.QMessageBox.Information,
            )
            return
        entries: list[tuple[str, list[tuple[str, str]]]] = []
        # Process each uploaded CSV and aggregate entries.
        for p in self.uploaded_files:
            try:
                result = self.ensemble.predict_csv(p, topk=self.topk)
                entries.extend(self._prepare_entries(result, header=Path(p).name))
            except Exception as exc:
                entries.append((Path(p).name, [(f"Fehler", str(exc))]))
        if entries:
            self._show_result_dialog(entries)

    def _on_manual_clicked(self):
        """Run inference on the manual single-row input."""
        try:
            df = self._collect_manual_df()
            result = self.ensemble.predict_dataframe(df, topk=self.topk)
            entries = self._prepare_entries(result)
            self._show_result_dialog(entries)
        except Exception as exc:
            self._show_message("Fehler", str(exc), QtWidgets.QMessageBox.Critical)


def main():
    """Launch the Qt application."""
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
