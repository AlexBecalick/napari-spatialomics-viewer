"""The "Manual registration" tab: a Qt widget that emits signals only.

The widget shows the coarse-alignment controls, crops, landmark table, fit summary and the
result view of the manual-registration feature. It does not compute fits or touch files, and it
must not import the controller, napari or zarr. User actions are emitted as signals (see
``SIGNAL_NAMES``); setters called by the controller never emit.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from qtpy.QtCore import QItemSelectionModel, Qt, Signal
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .manual_registration_model import (
    CROP_MIN_SIZE_PX,
    CROP_SNAP_PX,
    DISPLAY_MODES,
    MIN_DRAWN_CROP_PX,
    ROLES,
    SAME_SECTION_FLOOR_UM,
    SERIAL_SECTION_FLOOR_HINT_UM,
    CoarseState,
    CropRect,
    FitSummary,
    ResidualRow,
    snap_crop,
    wrap_deg,
)

SIGNAL_NAMES: tuple[str, ...] = (
    "rotate_by",
    "rotation_set",
    "shift_by",
    "offset_set",
    "flip_horizontal",
    "flip_vertical",
    "reset_coarse",
    "adopt_fit",
    "display_mode_changed",
    "flick_requested",
    "auto_switch_changed",
    "landmark_mode_changed",
    "pair_selected",
    "pair_enabled_changed",
    "delete_pairs",
    "undo_requested",
    "preview_changed",
    "outlier_floor_changed",
    "crop_draw_requested",
    "crop_set",
    "save_requested",
    "finish_requested",
    "overlay_changed",
)

NO_SESSION_TEXT = (
    "No registration store loaded. Dataset loader ▸ Manual Registration ▸ Load image data"
)
WARNING_COLOUR = "#e0a030"
_GREY = QColor("#888888")
_ROTATE_STEPS = (-90.0, -10.0, -1.0, -0.1, 0.1, 1.0, 10.0, 90.0)
_ROTATE_LABELS = {
    -90.0: "−90",
    -10.0: "−10",
    -1.0: "−1",
    -0.1: "−0.1",
    0.1: "+0.1",
    1.0: "+1",
    10.0: "+10",
    90.0: "+90",
}
_SHIFT_STEPS_UM = (1.0, 10.0, 100.0, 1000.0)
_OVERLAY_LABELS = {
    "reference_landmarks": ("Reference landmarks", True),
    "registered_landmarks": ("Registered moving landmarks", True),
    "landmark_fit_positions": ("Landmark-fit positions", False),
}
_CROP_FIELDS = ("x0", "y0", "x1", "y1")


def _fmt(value: float | None) -> str:
    """Format a residual with two decimals; ``None`` and NaN become an en dash."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value:.2f}"


def _flags(row: ResidualRow, *, off_text: str) -> str:
    parts = []
    if row.outlier:
        parts.append("⚠ outlier")
    if row.outside_crop:
        parts.append("✂ outside crop")
    if not row.enabled:
        parts.append(off_text)
    return " ".join(parts)


class _AngleSpinBox(QDoubleSpinBox):
    """Angle spin box whose arrows wrap through +-180 degrees.

    The built-in wrapping jumps from 180 straight back to the range minimum, which the controller
    wraps to 180 again, so the arrows could never cross +-180. Stepping through ``wrap_deg``
    lands on the equivalent angle instead.
    """

    def stepBy(self, steps: int) -> None:  # noqa: N802 - Qt override
        self.setValue(wrap_deg(self.value() + steps * self.singleStep()))


class ManualRegistrationWidget(QWidget):
    """The "Manual registration" tab (signals only; setters never emit)."""

    rotate_by = Signal(float)
    rotation_set = Signal(float)
    shift_by = Signal(float, float)  # um, x right, y down
    offset_set = Signal(float, float)  # um
    flip_horizontal = Signal()
    flip_vertical = Signal()
    reset_coarse = Signal()
    adopt_fit = Signal()
    display_mode_changed = Signal(str)  # "reference" | "moving" | "both"
    flick_requested = Signal()
    auto_switch_changed = Signal(bool)
    landmark_mode_changed = Signal(bool)
    pair_selected = Signal(int)
    pair_enabled_changed = Signal(int, bool)
    delete_pairs = Signal(list)  # list[int], ascending
    undo_requested = Signal()
    preview_changed = Signal(bool)
    outlier_floor_changed = Signal(float)
    crop_draw_requested = Signal(str)  # role
    crop_set = Signal(str, object)  # role, CropRect | None
    save_requested = Signal()
    finish_requested = Signal()
    overlay_changed = Signal(str, bool)  # key, visible (result mode)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._suppress = False
        self._kind: str | None = None
        self._offset: tuple[float, float] = (0.0, 0.0)
        self._coarse_enabled = True
        self._finish_enabled = False
        self._finish_reason = ""
        self._busy = False
        self._preview_available = False
        self._crop_drawing: str | None = None
        self._pending: tuple[str | None, int | None] = (None, None)
        self._crop_shapes: dict[str, tuple[int, int] | None] = {r: None for r in ROLES}
        self._crop_values: dict[str, CropRect | None] = {r: None for r in ROLES}
        self._crop_outside: dict[str, int] = {r: 0 for r in ROLES}
        self._build()
        self.set_session(None, "")

    # ------------------------------------------------------------------ construction

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        self._scroll = scroll
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        layout.addWidget(self._build_status())
        layout.addWidget(self._build_display())

        self._editing_box = QWidget()
        editing = QVBoxLayout(self._editing_box)
        editing.setContentsMargins(0, 0, 0, 0)
        editing.setSpacing(8)
        editing.addWidget(self._build_crop())
        editing.addWidget(self._build_coarse())
        editing.addWidget(self._build_landmarks())
        editing.addWidget(self._build_preview())
        editing.addWidget(self._build_save_row())
        layout.addWidget(self._editing_box)

        layout.addWidget(self._build_result())
        layout.addStretch(1)

    def _build_status(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        self._title_label = QLabel()
        font = self._title_label.font()
        font.setBold(True)
        self._title_label.setFont(font)
        self._title_label.setWordWrap(True)
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._prompt_label = QLabel()
        self._prompt_label.setWordWrap(True)
        self._busy_label = QLabel()
        self._busy_label.setVisible(False)
        for label in (self._title_label, self._status_label, self._prompt_label, self._busy_label):
            lay.addWidget(label)
        return box

    def _build_display(self) -> QWidget:
        self._display_box = QGroupBox("Display")
        lay = QVBoxLayout(self._display_box)
        row = QHBoxLayout()
        self._display_group = QButtonGroup(self)
        self._display_group.setExclusive(True)
        self._display_buttons: dict[str, QPushButton] = {}
        for mode, text in (("reference", "Reference"), ("moving", "Moving"), ("both", "Both")):
            button = QPushButton(text)
            button.setCheckable(True)
            self._display_group.addButton(button)
            self._display_buttons[mode] = button
            row.addWidget(button)
            button.toggled.connect(lambda checked, m=mode: self._on_display_toggled(m, checked))
        self._display_buttons["both"].setChecked(True)
        lay.addLayout(row)
        self._flick_button = QPushButton("Flick (F)")
        self._flick_button.setToolTip("F flicks between the images; hold Shift+F to peek")
        self._flick_button.clicked.connect(self._on_flick)
        lay.addWidget(self._flick_button)
        self._auto_switch_check = QCheckBox("Auto-switch after each click")
        self._auto_switch_check.setChecked(True)
        self._auto_switch_check.toggled.connect(self._on_auto_switch)
        lay.addWidget(self._auto_switch_check)
        return self._display_box

    def _build_crop(self) -> QWidget:
        self._crop_box = QGroupBox("Crop")
        lay = QVBoxLayout(self._crop_box)
        self._crop_draw_buttons: dict[str, QPushButton] = {}
        self._crop_clear_buttons: dict[str, QPushButton] = {}
        self._crop_spins: dict[str, dict[str, QSpinBox]] = {}
        self._crop_labels: dict[str, QLabel] = {}
        self._crop_outside_labels: dict[str, QLabel] = {}
        for role in ROLES:
            lay.addWidget(QLabel(f"{role.capitalize()} crop"))
            row = QHBoxLayout()
            draw = QPushButton("Draw")
            draw.clicked.connect(lambda _=False, r=role: self._on_crop_draw(r))
            self._crop_draw_buttons[role] = draw
            row.addWidget(draw)
            spins: dict[str, QSpinBox] = {}
            for name in _CROP_FIELDS:
                spin = QSpinBox()
                spin.setKeyboardTracking(False)
                spin.setRange(0, 1)
                spin.setPrefix(f"{name} ")
                spin.valueChanged.connect(lambda _v, r=role: self._on_crop_spins(r))
                spins[name] = spin
                row.addWidget(spin)
            self._crop_spins[role] = spins
            clear = QPushButton("Clear")
            clear.clicked.connect(lambda _=False, r=role: self._on_crop_clear(r))
            self._crop_clear_buttons[role] = clear
            row.addWidget(clear)
            lay.addLayout(row)
            label = QLabel()
            label.setWordWrap(True)
            self._crop_labels[role] = label
            lay.addWidget(label)
            outside = QLabel()
            outside.setStyleSheet(f"color: {WARNING_COLOUR}")
            self._crop_outside_labels[role] = outside
            lay.addWidget(outside)
        return self._crop_box

    def _build_coarse(self) -> QWidget:
        self._coarse_box = QGroupBox("Coarse alignment")
        lay = QVBoxLayout(self._coarse_box)

        lay.addWidget(QLabel("Rotation (counter-clockwise on screen)"))
        self._rotation_spin = _AngleSpinBox()
        self._rotation_spin.setRange(-180.0, 180.0)
        self._rotation_spin.setWrapping(True)
        self._rotation_spin.setDecimals(2)
        self._rotation_spin.setSingleStep(1.0)
        self._rotation_spin.setSuffix("°")
        self._rotation_spin.setKeyboardTracking(False)
        self._rotation_spin.valueChanged.connect(self._on_rotation_spin)
        lay.addWidget(self._rotation_spin)
        row = QHBoxLayout()
        self._rotate_buttons: dict[float, QPushButton] = {}
        for step in _ROTATE_STEPS:
            button = QPushButton(_ROTATE_LABELS[step])
            button.clicked.connect(lambda _=False, s=step: self._on_rotate(s))
            self._rotate_buttons[step] = button
            row.addWidget(button)
        lay.addLayout(row)
        self._rotate_180_button = QPushButton("180°")
        self._rotate_180_button.clicked.connect(lambda _=False: self._on_rotate(180.0))
        lay.addWidget(self._rotate_180_button)

        lay.addWidget(QLabel("Offset of the moving centre from the reference centre"))
        offset_row = QHBoxLayout()
        self._offset_x_spin = self._make_offset_spin("x ")
        self._offset_y_spin = self._make_offset_spin("y ")
        self._offset_x_spin.valueChanged.connect(self._on_offset_x)
        self._offset_y_spin.valueChanged.connect(self._on_offset_y)
        offset_row.addWidget(self._offset_x_spin)
        offset_row.addWidget(self._offset_y_spin)
        lay.addLayout(offset_row)

        grid = QGridLayout()
        self._shift_buttons: dict[str, QPushButton] = {}
        for key, text, (r, c) in (
            ("up", "↑", (0, 1)),
            ("left", "←", (1, 0)),
            ("right", "→", (1, 2)),
            ("down", "↓", (2, 1)),
        ):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, k=key: self._on_shift(k))
            self._shift_buttons[key] = button
            grid.addWidget(button, r, c)
        self._shift_step_combo = QComboBox()
        for step in _SHIFT_STEPS_UM:
            self._shift_step_combo.addItem(f"{step:g} µm", step)
        self._shift_step_combo.setCurrentIndex(_SHIFT_STEPS_UM.index(100.0))
        grid.addWidget(self._shift_step_combo, 1, 1)
        lay.addLayout(grid)

        flip_row = QHBoxLayout()
        self._flip_h_button = QPushButton("Flip H")
        self._flip_h_button.clicked.connect(lambda _=False: self._emit_plain(self.flip_horizontal))
        self._flip_v_button = QPushButton("Flip V")
        self._flip_v_button.clicked.connect(lambda _=False: self._emit_plain(self.flip_vertical))
        self._reset_button = QPushButton("Reset")
        self._reset_button.clicked.connect(lambda _=False: self._emit_plain(self.reset_coarse))
        for button in (self._flip_h_button, self._flip_v_button, self._reset_button):
            flip_row.addWidget(button)
        lay.addLayout(flip_row)
        self._mirror_label = QLabel("Mirrored: no")
        lay.addWidget(self._mirror_label)
        self._adopt_button = QPushButton("Use fit as coarse alignment")
        self._adopt_button.setToolTip("needs at least 3 enabled pairs")
        self._adopt_button.clicked.connect(lambda _=False: self._emit_plain(self.adopt_fit))
        lay.addWidget(self._adopt_button)
        return self._coarse_box

    @staticmethod
    def _make_offset_spin(prefix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1e6, 1e6)
        spin.setDecimals(2)
        spin.setSuffix(" µm")
        spin.setPrefix(prefix)
        spin.setKeyboardTracking(False)
        return spin

    def _build_landmarks(self) -> QWidget:
        self._landmark_box = QGroupBox("Landmarks")
        lay = QVBoxLayout(self._landmark_box)
        self._landmark_button = QPushButton("Landmark mode: off")
        self._landmark_button.setCheckable(True)
        self._landmark_button.toggled.connect(self._on_landmark_toggled)
        lay.addWidget(self._landmark_button)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(["#", "", "On", "Fit µm", "LOO µm", "Flags"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(160)
        self._table.cellClicked.connect(self._on_table_clicked)
        self._table.itemChanged.connect(self._on_table_item_changed)
        self._table.itemSelectionChanged.connect(self._update_delete_enabled)
        lay.addWidget(self._table)

        row = QHBoxLayout()
        self._undo_button = QPushButton("Undo")
        self._undo_button.setToolTip("Ctrl+Z")
        self._undo_button.clicked.connect(lambda _=False: self._emit_plain(self.undo_requested))
        self._delete_button = QPushButton("Delete selected")
        self._delete_button.setEnabled(False)
        self._delete_button.clicked.connect(self._on_delete)
        row.addWidget(self._undo_button)
        row.addWidget(self._delete_button)
        lay.addLayout(row)

        floor_row = QHBoxLayout()
        floor_row.addWidget(QLabel("Outlier floor"))
        self._floor_spin = QDoubleSpinBox()
        self._floor_spin.setRange(0.5, 1000.0)
        self._floor_spin.setDecimals(1)
        self._floor_spin.setSuffix(" µm")
        self._floor_spin.setKeyboardTracking(False)
        self._floor_spin.setValue(SAME_SECTION_FLOOR_UM)
        self._floor_spin.valueChanged.connect(self._on_floor_spin)
        floor_row.addWidget(self._floor_spin)
        self._floor_preset_buttons: dict[str, QPushButton] = {}
        for key, text, value in (
            ("same_section", "same (10 µm)", SAME_SECTION_FLOOR_UM),
            ("serial_section", "serial (30 µm)", SERIAL_SECTION_FLOOR_HINT_UM),
        ):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, v=value: self._floor_spin.setValue(v))
            self._floor_preset_buttons[key] = button
            floor_row.addWidget(button)
        lay.addLayout(floor_row)
        self._floor_hint = QLabel(
            "Preview matches the register run only at the preset's floor (same 10 µm, serial 30 µm)"
        )
        self._floor_hint.setWordWrap(True)
        lay.addWidget(self._floor_hint)

        self._summary_label = QLabel("A fit needs at least 2 enabled pairs.")
        self._summary_label.setWordWrap(True)
        lay.addWidget(self._summary_label)
        self._warnings_label = QLabel()
        self._warnings_label.setWordWrap(True)
        self._warnings_label.setStyleSheet(f"color: {WARNING_COLOUR}")
        self._warnings_label.setVisible(False)
        lay.addWidget(self._warnings_label)
        return self._landmark_box

    def _build_preview(self) -> QWidget:
        self._preview_button = QPushButton("Preview fit")
        self._preview_button.setCheckable(True)
        self._preview_button.setEnabled(False)
        self._preview_button.setToolTip("needs at least 3 enabled pairs")
        self._preview_button.toggled.connect(self._on_preview_toggled)
        return self._preview_button

    def _build_save_row(self) -> QWidget:
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        self._save_button = QPushButton("Save draft")
        self._save_button.clicked.connect(lambda _=False: self._emit_plain(self.save_requested))
        self._finish_button = QPushButton("Finish labelling")
        self._finish_button.setEnabled(False)
        self._finish_button.clicked.connect(lambda _=False: self._emit_plain(self.finish_requested))
        row.addWidget(self._save_button)
        row.addWidget(self._finish_button)
        return box

    def _build_result(self) -> QWidget:
        self._result_box = QGroupBox("Result")
        lay = QVBoxLayout(self._result_box)
        self._result_banner = QLabel()
        self._result_banner.setWordWrap(True)
        self._result_banner.setTextFormat(Qt.PlainText)
        lay.addWidget(self._result_banner)
        self._result_table = QTableWidget(0, 7)
        self._result_table.setHorizontalHeaderLabels(
            ["#", "", "Used", "Fit µm", "Affine µm", "Final µm", "Flags"]
        )
        self._result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._result_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._result_table.verticalHeader().setVisible(False)
        self._result_table.setMinimumHeight(160)
        self._result_table.cellClicked.connect(self._on_result_clicked)
        lay.addWidget(self._result_table)
        self._overlay_checks: dict[str, QCheckBox] = {}
        for key, (text, checked) in _OVERLAY_LABELS.items():
            check = QCheckBox(text)
            check.setChecked(checked)
            check.toggled.connect(lambda state, k=key: self._on_overlay(k, state))
            self._overlay_checks[key] = check
            lay.addWidget(check)
        return self._result_box

    # ------------------------------------------------------------------ helpers

    @contextmanager
    def _quiet(self) -> Iterator[None]:
        """Suppress handlers while a setter changes widgets programmatically."""
        previous = self._suppress
        self._suppress = True
        try:
            yield
        finally:
            self._suppress = previous

    def _emit_plain(self, signal) -> None:  # noqa: ANN001 (a bound Qt signal)
        if self._suppress:
            return
        signal.emit()

    def _refresh_enabled(self) -> None:
        """Recompute every enabled state from the session, busy, coarse and preview flags."""
        prepared = self._kind == "prepared"
        loaded = self._kind is not None
        editable = prepared and not self._busy
        self._display_box.setEnabled(loaded)
        self._editing_box.setEnabled(prepared)
        self._result_box.setEnabled(self._kind == "result")
        self._crop_box.setEnabled(editable)
        self._coarse_box.setEnabled(editable)
        self._landmark_box.setEnabled(editable)
        coarse = editable and self._coarse_enabled
        for widget in (
            self._rotation_spin,
            self._rotate_180_button,
            self._offset_x_spin,
            self._offset_y_spin,
            self._shift_step_combo,
            self._flip_h_button,
            self._flip_v_button,
            self._reset_button,
            *self._rotate_buttons.values(),
            *self._shift_buttons.values(),
        ):
            widget.setEnabled(coarse)
        self._adopt_button.setEnabled(coarse and self._preview_available)
        self._preview_button.setEnabled(editable and self._preview_available)
        self._save_button.setEnabled(editable)
        self._finish_button.setEnabled(editable and self._finish_enabled)
        for role in ROLES:
            drawing_other = self._crop_drawing is not None and self._crop_drawing != role
            self._crop_draw_buttons[role].setEnabled(not drawing_other)
        self._update_delete_enabled()

    def _update_delete_enabled(self) -> None:
        self._delete_button.setEnabled(bool(self._selected_ids()))

    def _selected_ids(self) -> list[int]:
        rows = {index.row() for index in self._table.selectedIndexes()}
        ids = []
        for row in rows:
            item = self._table.item(row, 0)
            if item is not None:
                ids.append(int(item.data(Qt.UserRole)))
        return sorted(ids)

    def _update_prompt(self) -> None:
        role, next_id = self._pending
        if role == "reference":
            text = f"Click the REFERENCE point of pair {next_id}"
        elif role == "moving":
            text = f"Pair {next_id}: now click the matching MOVING point (F flicks)"
        elif self._landmark_button.isChecked():
            text = "Landmark mode is on"
        else:
            text = ""
        self._prompt_label.setText(text)

    def _update_crop_labels(self, role: str, *, invalid: bool = False) -> None:
        label = self._crop_labels[role]
        shape = self._crop_shapes[role]
        crop = self._crop_values[role]
        label.setStyleSheet("")
        if invalid:
            label.setText(
                f"Invalid crop: x0 must be < x1 and y0 < y1, at least {MIN_DRAWN_CROP_PX} px"
            )
            label.setStyleSheet(f"color: {WARNING_COLOUR}")
        elif shape is None:
            label.setText("")
        elif crop is None:
            label.setText(f"Full image ({shape[1]} × {shape[0]} px)")
        else:
            text = f"{crop.x0}–{crop.x1} × {crop.y0}–{crop.y1} ({crop.width} × {crop.height} px)"
            try:
                eff = snap_crop(crop, shape)
                text += f" · effective {eff.width} × {eff.height} px"
            except ValueError:
                text += (
                    f" · too small for the workflow (min {CROP_MIN_SIZE_PX} px per side "
                    f"after {CROP_SNAP_PX}-px snapping)"
                )
                label.setStyleSheet(f"color: {WARNING_COLOUR}")
            label.setText(text)
        n = self._crop_outside[role]
        self._crop_outside_labels[role].setText(f"{n} pair(s) outside this crop ✂" if n else "")

    # ------------------------------------------------------------------ setters

    def set_session(self, kind: str | None, title: str) -> None:
        """Select the mode: ``None`` (nothing loaded), ``"prepared"`` or ``"result"``."""
        if kind not in (None, "prepared", "result"):
            raise ValueError(f"unknown session kind {kind!r}")
        with self._quiet():
            changed = kind != self._kind
            self._kind = kind
            if changed:
                self._landmark_button.blockSignals(True)
                self._landmark_button.setChecked(False)
                self._landmark_button.setText("Landmark mode: off")
                self._landmark_button.blockSignals(False)
                self._preview_button.blockSignals(True)
                self._preview_button.setChecked(False)
                self._preview_button.blockSignals(False)
                self._display_buttons["both"].setChecked(True)
                self._crop_drawing = None
                for role in ROLES:
                    self._crop_draw_buttons[role].setText("Draw")
                self._pending = (None, None)
                self._busy = False
                self._busy_label.setText("")
                self._busy_label.setVisible(False)
                self._set_rows_impl([])
                self._set_summary_impl(None)
                self._result_table.setRowCount(0)
                self._result_banner.setText("")
            self._display_buttons["moving"].setText(
                "Registered moving" if kind == "result" else "Moving"
            )
            self._editing_box.setVisible(kind != "result")
            self._result_box.setVisible(kind == "result")
            if kind is None:
                self._title_label.setText("")
                self._status_label.setText(NO_SESSION_TEXT)
            else:
                self._title_label.setText(title)
                if changed:
                    self._status_label.setText("")
            self._update_prompt()
            self._refresh_enabled()

    def set_coarse(self, coarse: CoarseState) -> None:
        """Show the coarse state (spin boxes and mirror label); remembers the exact offset."""
        with self._quiet():
            for spin in (self._rotation_spin, self._offset_x_spin, self._offset_y_spin):
                spin.blockSignals(True)
            try:
                self._rotation_spin.setValue(coarse.rotation_deg)
                self._offset_x_spin.setValue(coarse.offset_um[0])
                self._offset_y_spin.setValue(coarse.offset_um[1])
            finally:
                for spin in (self._rotation_spin, self._offset_x_spin, self._offset_y_spin):
                    spin.blockSignals(False)
            self._offset = (float(coarse.offset_um[0]), float(coarse.offset_um[1]))
            self._mirror_label.setText("Mirrored: yes" if coarse.mirror_x else "Mirrored: no")

    def set_coarse_enabled(self, enabled: bool) -> None:
        """Enable or disable the coarse-alignment controls (rotation, offset, flips, reset)."""
        with self._quiet():
            self._coarse_enabled = bool(enabled)
            self._refresh_enabled()

    def set_display_mode(self, mode: str) -> None:
        """Check the display button of ``mode``; ``ValueError`` on an unknown mode."""
        if mode not in DISPLAY_MODES:
            raise ValueError(f"unknown display mode {mode!r}")
        with self._quiet():
            self._display_buttons[mode].setChecked(True)

    def set_landmark_mode(self, active: bool) -> None:
        """Show whether landmark mode is on."""
        with self._quiet():
            self._landmark_button.blockSignals(True)
            self._landmark_button.setChecked(bool(active))
            self._landmark_button.blockSignals(False)
            self._landmark_button.setText(f"Landmark mode: {'on' if active else 'off'}")
            self._update_prompt()

    def set_pending(self, role: str | None, next_id: int | None) -> None:
        """Show where the next click goes (``role`` is the image of the next click)."""
        with self._quiet():
            self._pending = (role, next_id)
            self._update_prompt()

    def _set_rows_impl(self, rows: Sequence[ResidualRow]) -> None:
        selected = set(self._selected_ids())
        table = self._table
        table.blockSignals(True)
        try:
            table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                id_item = QTableWidgetItem(str(row.id))
                id_item.setData(Qt.UserRole, int(row.id))
                swatch = QTableWidgetItem("")
                swatch.setBackground(QBrush(QColor(row.colour)))
                on_item = QTableWidgetItem("")
                on_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                on_item.setCheckState(Qt.Checked if row.enabled else Qt.Unchecked)
                items = [
                    id_item,
                    swatch,
                    on_item,
                    QTableWidgetItem(_fmt(row.fit_um)),
                    QTableWidgetItem(_fmt(row.loo_um)),
                    QTableWidgetItem(_flags(row, off_text="off")),
                ]
                for c, item in enumerate(items):
                    if c != 2:
                        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    if not row.enabled and c != 1:
                        item.setForeground(QBrush(_GREY))
                    table.setItem(r, c, item)
            table.clearSelection()
            for r, row in enumerate(rows):
                if row.id in selected:
                    # selectRow would clear the earlier rows under ExtendedSelection
                    table.selectionModel().select(
                        table.model().index(r, 0),
                        QItemSelectionModel.Select | QItemSelectionModel.Rows,
                    )
        finally:
            table.blockSignals(False)
        self._update_delete_enabled()

    def set_rows(self, rows: Sequence[ResidualRow]) -> None:
        """Fill the landmark table (the selection is kept by pair id)."""
        with self._quiet():
            self._set_rows_impl(rows)

    def _set_summary_impl(self, summary: FitSummary | None) -> None:
        if summary is None:
            self._summary_label.setText("A fit needs at least 2 enabled pairs.")
            self._warnings_label.setText("")
            self._warnings_label.setVisible(False)
            return
        self._summary_label.setText(
            f"Fit on {summary.n_used} of {summary.n_enabled} enabled pairs · "
            f"RMS {summary.rms_um:.2f} µm · scale ratio {summary.scale_ratio:.4f} · "
            f"rotation {summary.rotation_deg:.2f}° · "
            f"mirrored {'yes' if summary.reflected else 'no'} · "
            f"spread {summary.spread_ratio:.2f}"
        )
        self._warnings_label.setText("\n".join(summary.warnings))
        self._warnings_label.setVisible(bool(summary.warnings))

    def set_fit_summary(self, summary: FitSummary | None) -> None:
        """Show the fit summary and its warnings (``None``: not enough pairs)."""
        with self._quiet():
            self._set_summary_impl(summary)

    def set_crop(
        self,
        role: str,
        crop: CropRect | None,
        shape_yx: tuple[int, int],
        n_pairs_outside: int,
    ) -> None:
        """Show the crop of ``role`` (``None``: full image) and how many pairs lie outside it."""
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        height, width = shape_yx
        with self._quiet():
            self._crop_shapes[role] = (int(height), int(width))
            self._crop_values[role] = crop
            self._crop_outside[role] = int(n_pairs_outside)
            spins = self._crop_spins[role]
            ranges = {
                "x0": (0, width - 1),
                "x1": (1, width),
                "y0": (0, height - 1),
                "y1": (1, height),
            }
            values = (
                {"x0": 0, "y0": 0, "x1": width, "y1": height}
                if crop is None
                else {"x0": crop.x0, "y0": crop.y0, "x1": crop.x1, "y1": crop.y1}
            )
            for name in _CROP_FIELDS:
                spin = spins[name]
                spin.blockSignals(True)
                spin.setRange(*ranges[name])
                spin.setValue(values[name])
                spin.blockSignals(False)
            self._update_crop_labels(role)

    def set_crop_drawing(self, role: str | None) -> None:
        """Reflect the armed crop draw (``None``: not drawing)."""
        if role is not None and role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        with self._quiet():
            self._crop_drawing = role
            for r in ROLES:
                self._crop_draw_buttons[r].setText(
                    "Drawing… (Esc cancels)" if r == role else "Draw"
                )
            self._refresh_enabled()

    def set_preview_available(self, available: bool) -> None:
        """Enable the Preview toggle and "Use fit" (the checked state is unchanged)."""
        with self._quiet():
            self._preview_available = bool(available)
            self._refresh_enabled()

    def set_preview(self, active: bool) -> None:
        """Show whether the fit preview is on."""
        with self._quiet():
            self._preview_button.blockSignals(True)
            self._preview_button.setChecked(bool(active))
            self._preview_button.blockSignals(False)

    def set_finish_state(self, enabled: bool, reason: str) -> None:
        """Enable "Finish labelling"; ``reason`` is the tooltip (why it is disabled)."""
        with self._quiet():
            self._finish_enabled = bool(enabled)
            self._finish_reason = reason
            self._finish_button.setToolTip(reason or "Finish labelling and save")
            self._refresh_enabled()

    def set_dirty(self, dirty: bool, status_text: str) -> None:
        """Show the status line; unsaved changes add a leading marker."""
        with self._quiet():
            prefix = "● unsaved changes · " if dirty else ""
            self._status_label.setText(prefix + status_text)

    def set_busy(self, busy: bool) -> None:
        """Disable saving, crops, coarse and landmark sections while a save runs."""
        with self._quiet():
            self._busy = bool(busy)
            self._busy_label.setText("Saving…" if busy else "")
            self._busy_label.setVisible(bool(busy))
            self._refresh_enabled()

    def set_result(self, rows: Sequence[ResidualRow], banner_text: str) -> None:
        """Fill the result table and banner (result mode)."""
        with self._quiet():
            self._result_banner.setText(banner_text)
            table = self._result_table
            table.blockSignals(True)
            try:
                table.setRowCount(len(rows))
                for r, row in enumerate(rows):
                    id_item = QTableWidgetItem(str(row.id))
                    id_item.setData(Qt.UserRole, int(row.id))
                    swatch = QTableWidgetItem("")
                    swatch.setBackground(QBrush(QColor(row.colour)))
                    items = [
                        id_item,
                        swatch,
                        QTableWidgetItem("yes" if row.used else "no"),
                        QTableWidgetItem(_fmt(row.fit_um)),
                        QTableWidgetItem(_fmt(row.affine_um)),
                        QTableWidgetItem(_fmt(row.final_um)),
                        QTableWidgetItem(_flags(row, off_text="held out")),
                    ]
                    for c, item in enumerate(items):
                        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                        if not row.enabled and c != 1:
                            item.setForeground(QBrush(_GREY))
                        table.setItem(r, c, item)
            finally:
                table.blockSignals(False)

    def set_overlay_visibility(self, key: str, visible: bool) -> None:
        """Show the state of a result-mode overlay checkbox."""
        if key not in self._overlay_checks:
            raise ValueError(f"unknown overlay {key!r}")
        with self._quiet():
            check = self._overlay_checks[key]
            check.blockSignals(True)
            check.setChecked(bool(visible))
            check.blockSignals(False)

    def set_outlier_floor(self, value_um: float) -> None:
        """Set the outlier-floor spin box without emitting."""
        with self._quiet():
            self._floor_spin.blockSignals(True)
            self._floor_spin.setValue(float(value_um))
            self._floor_spin.blockSignals(False)

    # ------------------------------------------------------------------ getters

    def display_mode(self) -> str:
        """The checked display mode."""
        for mode, button in self._display_buttons.items():
            if button.isChecked():
                return mode
        return "both"

    def auto_switch(self) -> bool:
        """Whether auto-switch after each click is on."""
        return self._auto_switch_check.isChecked()

    def outlier_floor_um(self) -> float:
        """The outlier floor in the spin box (micrometres)."""
        return float(self._floor_spin.value())

    def shift_step_um(self) -> float:
        """The selected arrow-key shift step (micrometres)."""
        return float(self._shift_step_combo.currentData())

    def landmark_mode(self) -> bool:
        """Whether landmark mode is on."""
        return self._landmark_button.isChecked()

    def preview(self) -> bool:
        """Whether the fit preview is on."""
        return self._preview_button.isChecked()

    # ------------------------------------------------------------------ handlers

    def _on_display_toggled(self, mode: str, checked: bool) -> None:
        if self._suppress or not checked:
            return
        self.display_mode_changed.emit(mode)

    def _on_flick(self) -> None:
        if self._suppress:
            return
        self.flick_requested.emit()

    def _on_auto_switch(self, checked: bool) -> None:
        if self._suppress:
            return
        self.auto_switch_changed.emit(bool(checked))

    def _on_crop_draw(self, role: str) -> None:
        if self._suppress:
            return
        self.crop_draw_requested.emit(role)

    def _on_crop_clear(self, role: str) -> None:
        if self._suppress:
            return
        self.crop_set.emit(role, None)

    def _on_crop_spins(self, role: str) -> None:
        if self._suppress:
            return
        shape = self._crop_shapes[role]
        if shape is None:
            return
        height, width = shape
        spins = self._crop_spins[role]
        x0, y0, x1, y1 = (spins[name].value() for name in _CROP_FIELDS)
        if not (
            x0 < x1 and y0 < y1 and x1 - x0 >= MIN_DRAWN_CROP_PX and y1 - y0 >= MIN_DRAWN_CROP_PX
        ):
            self._update_crop_labels(role, invalid=True)
            return
        if (x0, y0, x1, y1) == (0, 0, width, height):
            self.crop_set.emit(role, None)
        else:
            self.crop_set.emit(role, CropRect(x0, y0, x1, y1))

    def _on_rotate(self, delta: float) -> None:
        if self._suppress:
            return
        self.rotate_by.emit(float(delta))

    def _on_rotation_spin(self, value: float) -> None:
        if self._suppress:
            return
        self.rotation_set.emit(float(value))

    def _on_offset_x(self, value: float) -> None:
        if self._suppress:
            return
        self.offset_set.emit(float(value), self._offset[1])

    def _on_offset_y(self, value: float) -> None:
        if self._suppress:
            return
        self.offset_set.emit(self._offset[0], float(value))

    def _on_shift(self, key: str) -> None:
        if self._suppress:
            return
        step = self.shift_step_um()
        dx, dy = {
            "left": (-step, 0.0),
            "right": (step, 0.0),
            "up": (0.0, -step),
            "down": (0.0, step),
        }[key]
        self.shift_by.emit(dx, dy)

    def _on_landmark_toggled(self, checked: bool) -> None:
        self._landmark_button.setText(f"Landmark mode: {'on' if checked else 'off'}")
        self._update_prompt()
        if self._suppress:
            return
        self.landmark_mode_changed.emit(bool(checked))

    def _on_table_clicked(self, row: int, column: int) -> None:
        if self._suppress or column == 2:
            return
        item = self._table.item(row, 0)
        if item is not None:
            self.pair_selected.emit(int(item.data(Qt.UserRole)))

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._suppress or item.column() != 2:
            return
        id_item = self._table.item(item.row(), 0)
        if id_item is not None:
            self.pair_enabled_changed.emit(
                int(id_item.data(Qt.UserRole)), item.checkState() == Qt.Checked
            )

    def _on_delete(self) -> None:
        if self._suppress:
            return
        ids = self._selected_ids()
        if ids:
            self.delete_pairs.emit(ids)

    def _on_floor_spin(self, value: float) -> None:
        if self._suppress:
            return
        self.outlier_floor_changed.emit(float(value))

    def _on_preview_toggled(self, checked: bool) -> None:
        if self._suppress:
            return
        self.preview_changed.emit(bool(checked))

    def _on_result_clicked(self, row: int, _column: int) -> None:
        if self._suppress:
            return
        item = self._result_table.item(row, 0)
        if item is not None:
            self.pair_selected.emit(int(item.data(Qt.UserRole)))

    def _on_overlay(self, key: str, checked: bool) -> None:
        if self._suppress:
            return
        self.overlay_changed.emit(key, bool(checked))
