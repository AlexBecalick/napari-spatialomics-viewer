"""Right-panel HITL crop drawing, estimates and portable region export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from napari.layers import Image
from napari.qt.threading import thread_worker
from qtpy.QtCore import QSettings, QTimer
from qtpy.QtWidgets import (
    QFileDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .hitl_crops import (
    CropCellCounts,
    HitlCropContext,
    count_crop_cells,
    save_crop_regions,
    validate_crop_rectangles,
)


@dataclass
class _CropSession:
    context: HitlCropContext
    rectangles: list[np.ndarray] = field(default_factory=list)


class HitlCropWidget(QWidget):
    """Select annotation regions without using the viewer's Python console."""

    def __init__(self, viewer: object, settings: QSettings | None = None) -> None:
        super().__init__()
        self._viewer = viewer
        self._settings = settings or QSettings(
            "Napari Compare Xenium MERSCOPE", "Spatialomics Viewer"
        )
        self._store: Path | None = None
        self._platform = ""
        self._sessions: dict[str, _CropSession] = {}
        self._layer = None
        self._guides = None
        self._updating = False
        self._drawing = False
        self._generation = 0
        self._worker = None
        self._pending_count = False
        self._counts: CropCellCounts | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._start_count)

        description = QLabel(
            "Choose regions here, then correct their cell outlines in the "
            "separate Cellpose drawing app."
        )
        description.setWordWrap(True)
        self._setup_button = QPushButton("Load crop setup…")
        self._setup_button.clicked.connect(self._choose_setup)
        self._add_button = QPushButton("Add rectangular crop")
        self._add_button.setCheckable(True)
        self._add_button.clicked.connect(self._arm_drawing)
        self._crop_list = QListWidget()
        self._crop_list.currentRowChanged.connect(self._select_crop)
        self._remove_button = QPushButton("Remove selected crop")
        self._remove_button.clicked.connect(self._remove_crop)
        self._counter = QLabel("Estimated selected cells: 0")
        self._counter.setWordWrap(True)
        self._target = QLabel("Aim for 800–1,200 cells")
        self._target.setWordWrap(True)
        self._status = QLabel("Load a standalone Atera or Xenium dataset.")
        self._status.setWordWrap(True)
        self._save_button = QPushButton("Save crop regions…")
        self._save_button.clicked.connect(self._save)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        for widget in (description, self._setup_button, self._add_button):
            layout.addWidget(widget)
        layout.addWidget(self._crop_list, stretch=1)
        for widget in (
            self._remove_button,
            self._counter,
            self._target,
            self._status,
            self._save_button,
        ):
            layout.addWidget(widget)
        self.set_dataset(None)

    def set_dataset(self, store: Path | None, platform: str = "") -> None:
        """Restore this store's selections and disable selection in paired views."""
        self._generation += 1
        self._timer.stop()
        self._layer = self._guides = None
        self._drawing = False
        self._add_button.setChecked(False)
        self._store = Path(store) if store is not None else None
        self._platform = str(platform)
        eligible = self._store is not None and platform in {"ATERA", "XENIUM"}
        self._setup_button.setEnabled(eligible)
        self._add_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._remove_button.setEnabled(False)
        self._crop_list.clear()
        self._set_count(None)
        if not eligible:
            self._status.setText(
                "Open an Atera or Xenium dataset standalone to choose HITL crops."
            )
            return
        if self._session is None:
            candidates = [
                self._store / "merxen_hitl_roi_context.json",
                self._store.parent / "roi_context.json",
            ]
            remembered = self._settings.value("hitl_crops/last_context", "", type=str)
            if remembered:
                candidates.append(Path(remembered))
            for candidate in candidates:
                if candidate.is_file():
                    try:
                        self.load_setup(candidate)
                        return
                    except (ValueError, TypeError, OSError):
                        continue
            self._status.setText(
                "Click Load crop setup and choose the roi_context.json "
                "from your annotation package."
            )
        else:
            self._restore_layers()

    @property
    def _session(self) -> _CropSession | None:
        return self._sessions.get(str(self._store)) if self._store is not None else None

    def load_setup(self, path: Path) -> None:
        """Load and bind a portable crop setup to the active native source."""
        if self._store is None:
            raise ValueError("Open a standalone dataset first.")
        context = HitlCropContext.read(path)
        context.validate_store(self._store, self._platform)
        previous = self._session
        if previous is not None and previous.context != context and previous.rectangles:
            raise ValueError(
                "Save your current regions before loading a different crop setup."
            )
        self._sessions[str(self._store)] = (
            previous
            if previous is not None and previous.context == context
            else _CropSession(context)
        )
        self._settings.setValue("hitl_crops/last_context", str(path))
        self._restore_layers()

    def _choose_setup(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load HITL crop setup", "", "Crop setup (*.json)"
        )
        if path:
            try:
                self.load_setup(Path(path))
            except (ValueError, TypeError, OSError) as error:
                QMessageBox.warning(self, "HITL seg crops", str(error))

    def _restore_layers(self) -> None:
        session = self._session
        if session is None:
            return
        for name in ("HITL crop regions", "HITL valid crop areas"):
            if name in self._viewer.layers:
                self._viewer.layers.remove(name)
        self._guides = self._viewer.add_shapes(
            [area.vertices() for area in session.context.usable_areas()],
            name="HITL valid crop areas",
            ndim=2,
            shape_type="rectangle",
            scale=(session.context.pixel_size_um,) * 2,
            face_color="transparent",
            edge_color="cyan",
            opacity=0.35,
            edge_width=2,
        )
        self._guides.editable = False
        self._layer = self._viewer.add_shapes(
            session.rectangles,
            name="HITL crop regions",
            ndim=2,
            shape_type="rectangle",
            scale=(session.context.pixel_size_um,) * 2,
            face_color="transparent",
            edge_color="yellow",
            edge_width=3,
            metadata={"hitl_seg_crops": True},
        )
        self._layer.events.data.connect(self._on_data)
        self._add_button.setEnabled(True)
        self._status.setText(
            "Click Add rectangular crop, then click and drag. "
            "Keep each crop inside one cyan guide."
        )
        self._sync_selection()

    def _require_native_image(self) -> None:
        context = self._session.context
        for layer in self._viewer.layers:
            if (
                not isinstance(layer, Image)
                or context.image_key not in layer.name
                or context.platform not in layer.name.upper()
                or "aligned" in layer.name.lower()
            ):
                continue
            data = layer.data[0] if layer.multiscale else layer.data
            shape = np.asarray(data.shape[-3:-1] if layer.rgb else data.shape[-2:])
            if layer.ndim != 2:
                continue
            expected = (
                np.asarray([context.height, context.width])
                / shape
                * context.pixel_size_um
            )
            origin = np.asarray(layer.data_to_world((0, 0))[-2:])
            matrix = np.column_stack(
                [
                    np.asarray(layer.data_to_world(point)[-2:]) - origin
                    for point in ((1, 0), (0, 1))
                ]
            )
            if np.all(
                abs(matrix - np.diag(expected)) <= np.diag(expected / shape) + 1e-8
            ) and np.all(abs(origin) <= expected):
                return
        raise ValueError(
            "Load the native morphology image from the Images tab. "
            "Crops require a standalone, unshifted micron view."
        )

    def _arm_drawing(self, checked: bool = True) -> None:
        try:
            self._require_native_image()
            if self._layer not in self._viewer.layers:
                self._restore_layers()
            self._drawing = bool(checked)
            self._layer.mode = "add_rectangle" if checked else "select"
            self._viewer.layers.selection.active = self._layer
            self._status.setText(
                "Click and drag to draw one crop. "
                "Press Add rectangular crop again for the next crop."
            )
        except ValueError as error:
            self._add_button.setChecked(False)
            self._status.setText(str(error))

    def _on_data(self, event: object) -> None:
        if self._updating:
            return
        action = getattr(
            getattr(event, "action", ""), "value", str(getattr(event, "action", ""))
        )
        if action in {"adding", "changing", "removing"}:
            return
        # Napari emits ADDED before completing its drag. Changing mode inside
        # that signal re-enters the drawing routine; defer until release ends.
        QTimer.singleShot(0, self._finish_change)

    def _finish_change(self) -> None:
        if self._layer is None or self._layer not in self._viewer.layers:
            return
        if self._drawing:
            self._drawing = False
            self._add_button.setChecked(False)
            self._layer.mode = "select"
        self._sync_selection()

    def _sync_selection(self) -> None:
        session = self._session
        if session is None or self._layer is None:
            return
        session.rectangles = [np.rint(points).copy() for points in self._layer.data]
        if any(
            not np.array_equal(points, rounded)
            for points, rounded in zip(
                self._layer.data, session.rectangles, strict=True
            )
        ):
            self._updating = True
            try:
                self._layer.data = session.rectangles
            finally:
                self._updating = False
        self._generation += 1
        self._counts = None
        self._render_list()
        self._set_count(None)
        self._remove_button.setEnabled(bool(session.rectangles))
        try:
            for points in session.rectangles:
                session.context.rectangle(points)
        except ValueError as error:
            self._save_button.setEnabled(False)
            self._status.setText(str(error))
            return
        try:
            validate_crop_rectangles(session.context, session.rectangles)
        except ValueError as error:
            self._save_button.setEnabled(False)
            self._status.setText(str(error))
        else:
            self._save_button.setEnabled(bool(session.rectangles))
            self._status.setText(
                "Move or resize crops with the selection tool. "
                "Correct every visible cell in each exported crop."
            )
        if session.rectangles:
            self._timer.start()
        else:
            self._set_count(CropCellCounts((), 0, ""))

    def _render_list(self) -> None:
        row = self._crop_list.currentRow()
        self._crop_list.blockSignals(True)
        self._crop_list.clear()
        for index, points in enumerate(self._session.rectangles):
            h, w = np.ptp(points, axis=0) * self._session.context.pixel_size_um
            count = (
                f" · ~{self._counts.per_crop[index]} cells"
                if self._counts is not None
                else ""
            )
            self._crop_list.addItem(f"Crop {index + 1} · {w:.0f} × {h:.0f} µm{count}")
        if self._crop_list.count():
            self._crop_list.setCurrentRow(min(max(row, 0), self._crop_list.count() - 1))
        self._crop_list.blockSignals(False)

    def _select_crop(self, row: int) -> None:
        if self._layer is not None and 0 <= row < len(self._layer.data):
            self._viewer.layers.selection.active = self._layer
            self._layer.selected_data = {row}

    def _remove_crop(self) -> None:
        row = self._crop_list.currentRow()
        if self._layer is not None and 0 <= row < len(self._layer.data):
            self._layer.selected_data = {row}
            self._layer.remove_selected()

    def _start_count(self) -> None:
        if self._worker is not None:
            self._pending_count = True
            return
        session = self._session
        if session is None or not session.rectangles:
            return
        try:
            rectangles = [
                session.context.rectangle(points) for points in session.rectangles
            ]
        except ValueError:
            return
        generation = self._generation
        store, context = self._store, session.context
        self._counter.setText("Estimating selected cells…")

        def compute() -> CropCellCounts:
            return count_crop_cells(store, context, rectangles)

        worker = thread_worker(compute)()
        worker.returned.connect(lambda result: self._count_done(generation, result))
        worker.errored.connect(lambda error: self._count_failed(generation, error))
        worker.finished.connect(self._count_finished)
        self._worker = worker
        worker.start()

    def _count_done(self, generation: int, result: CropCellCounts) -> None:
        if generation == self._generation:
            self._counts = result
            self._set_count(result)
            self._render_list()

    def _count_failed(self, generation: int, error: Exception) -> None:
        if generation == self._generation:
            self._set_count(None)
            self._counter.setText(f"Cell estimate unavailable: {error}")

    def _count_finished(self) -> None:
        self._worker = None
        if self._pending_count:
            self._pending_count = False
            self._timer.start()

    def _set_count(self, counts: CropCellCounts | None) -> None:
        total = counts.total if counts is not None else None
        self._counter.setText(
            "Estimated selected cells: —"
            if total is None
            else f"Estimated selected cells: {total:,}"
        )
        reached = total is not None and 800 <= total <= 1200
        self._target.setProperty("targetReached", reached)
        self._target.setText(
            "800–1,200 cells: target reached" if reached else "Aim for 800–1,200 cells"
        )
        self._target.setStyleSheet(
            "background: #21633d; color: white; padding: 6px; border-radius: 4px;"
            if reached
            else ""
        )

    def _save(self) -> None:
        session = self._session
        if session is None:
            return
        try:
            self._require_native_image()
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save HITL crop regions",
                "hitl-regions.geojson",
                "GeoJSON (*.geojson)",
            )
            if not path:
                return
            target = Path(path)
            if not target.suffix:
                target = target.with_suffix(".geojson")
            if target.resolve().is_relative_to(self._store.resolve()):
                raise ValueError("Save regions outside the source store.")
            save_crop_regions(target, session.context, session.rectangles)
            self._status.setText(
                f"Saved {len(session.rectangles)} crop regions. "
                "Export these regions into a Cellpose annotation package."
            )
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "HITL seg crops", str(error))
