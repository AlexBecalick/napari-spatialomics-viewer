"""Right-panel HITL crop drawing, estimates and portable region export."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
from napari.layers import Image
from napari.qt.threading import thread_worker
from qtpy.QtCore import QSettings, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QFileDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .hitl_annotation_session import (
    AnnotationPackage,
    annotation_progress,
    open_annotation_package,
    validate_saved_annotation,
)
from .hitl_crops import (
    CropCellCounts,
    HitlCropContext,
    count_crop_cells,
    save_crop_regions,
    validate_crop_rectangles,
)
from .hitl_package import (
    export_local_annotation_package,
    find_local_density,
    load_annotation_export_settings,
    upgrade_legacy_annotation_package,
    validate_local_density,
)


@dataclass
class _CropSession:
    context: HitlCropContext
    rectangles: list[np.ndarray] = field(default_factory=list)
    setup_path: Path | None = None
    package: AnnotationPackage | None = None


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
        self._offline_session: _CropSession | None = None
        self._layer = None
        self._guides = None
        self._updating = False
        self._drawing = False
        self._generation = 0
        self._worker = None
        self._pending_count = False
        self._counts: CropCellCounts | None = None
        self._export_worker = None
        self._annotation_launch = None
        self._gui_timer = QTimer(self)
        self._gui_timer.setInterval(500)
        self._gui_timer.timeout.connect(self._poll_annotation_gui)
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
        self._regions_button = QPushButton("Save regions only…")
        self._regions_button.clicked.connect(self._save)
        self._save_button = QPushButton("Save areas and prepare annotation")
        self._save_button.clicked.connect(self._prepare_annotation)
        self._annotate_button = QPushButton("Annotate cell masks")
        self._annotate_button.clicked.connect(self._annotate_selected_crop)
        self._open_package_button = QPushButton("Open annotation package…")
        self._open_package_button.clicked.connect(self._choose_annotation_package)
        self._python_button = QPushButton("Choose Cellpose environment…")
        self._python_button.clicked.connect(self._choose_cellpose_python)
        self._new_selection_button = QPushButton("Start new crop selection")
        self._new_selection_button.clicked.connect(self._new_crop_selection)
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
            self._annotate_button,
            self._open_package_button,
            self._python_button,
            self._new_selection_button,
            self._regions_button,
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
        self._annotate_button.setEnabled(False)
        self._regions_button.setEnabled(False)
        self._open_package_button.setEnabled(True)
        self._new_selection_button.setEnabled(eligible)
        self._remove_button.setEnabled(False)
        self._crop_list.clear()
        self._set_count(None)
        if not eligible:
            if self._store is None and self._offline_session is not None:
                self._update_annotation_state()
                return
            self._status.setText(
                "Open an Atera or Xenium dataset standalone to choose HITL crops."
            )
            return
        if self._session is None:
            remembered_package = self._settings.value(
                self._package_setting_key(), "", type=str
            )
            if remembered_package and Path(remembered_package).is_dir():
                try:
                    self.open_annotation_package(Path(remembered_package))
                    return
                except (OSError, ValueError, KeyError, TypeError):
                    pass
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
        return (
            self._sessions.get(str(self._store))
            if self._store is not None
            else self._offline_session
        )

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
        self._session.setup_path = path
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
        if self._store is None and session.package is not None:
            self._update_annotation_state()
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
        self._layer.editable = session.package is None
        self._add_button.setEnabled(session.package is None)
        self._status.setText(
            "Click Add rectangular crop, then click and drag. "
            "Keep each crop inside one cyan guide."
        )
        self._sync_selection()
        if self._export_worker is not None:
            self._layer.editable = False
            for button in (
                self._add_button,
                self._remove_button,
                self._save_button,
                self._regions_button,
            ):
                button.setEnabled(False)

    def _require_native_image(self) -> None:
        context = self._session.context
        found_native = False
        for layer in self._viewer.layers:
            if not isinstance(layer, Image):
                continue
            source = layer.metadata.get("spatialdata_image_source")
            if source is not None:
                if (
                    not isinstance(source, dict)
                    or source.get("image_key") != context.image_key
                    or source.get("platform") != context.platform
                    or source.get("coordinate_frame") != "native"
                    or not source.get("store_path")
                    or Path(source["store_path"]).resolve() != self._store.resolve()
                ):
                    continue
            elif (
                context.image_key not in layer.name
                or context.platform not in layer.name.upper()
                or "aligned" in layer.name.lower()
            ):
                # Keep the original programmatic layer naming convention;
                # normal viewer layers use their explicit backing identity.
                continue
            found_native = True
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
        if found_native:
            raise ValueError(
                "The morphology image is rotated, shifted or scaled away from "
                "the native crop grid. Reset the dataset rotation and image transform."
            )
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
        if session.package is not None:
            self._update_annotation_state()
            return
        self._remove_button.setEnabled(bool(session.rectangles))
        try:
            for points in session.rectangles:
                session.context.rectangle(points)
        except ValueError as error:
            self._save_button.setEnabled(False)
            self._regions_button.setEnabled(False)
            self._status.setText(str(error))
            return
        try:
            validate_crop_rectangles(session.context, session.rectangles)
        except ValueError as error:
            self._save_button.setEnabled(False)
            self._regions_button.setEnabled(False)
            self._status.setText(str(error))
        else:
            self._save_button.setEnabled(bool(session.rectangles))
            self._regions_button.setEnabled(bool(session.rectangles))
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
        package = self._session.package
        completed = annotation_progress(package) if package is not None else {}
        for index, points in enumerate(self._session.rectangles):
            h, w = np.ptp(points, axis=0) * self._session.context.pixel_size_um
            count = (
                f" · ~{self._counts.per_crop[index]} cells"
                if self._counts is not None
                else ""
            )
            status = ""
            if package is not None:
                crop = package.crops[index]
                if crop.crop_id in completed:
                    status = " · Annotated ✓"
                    count = f" · {completed[crop.crop_id].cell_count} cells"
            self._crop_list.addItem(
                f"Crop {index + 1} · {w:.0f} × {h:.0f} µm{count}{status}"
            )
            if status:
                self._crop_list.item(index).setForeground(QColor("#3ecf6b"))
        if self._crop_list.count():
            self._crop_list.setCurrentRow(min(max(row, 0), self._crop_list.count() - 1))
        self._crop_list.blockSignals(False)

    def _select_crop(self, row: int) -> None:
        if self._layer is not None and 0 <= row < len(self._layer.data):
            self._viewer.layers.selection.active = self._layer
            self._layer.selected_data = {row}
        self._annotate_button.setEnabled(
            self._session is not None
            and self._session.package is not None
            and row >= 0
            and self._annotation_launch is None
        )

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

    def _package_setting_key(self) -> str:
        key = hashlib.sha256(str(self._store).encode()).hexdigest()
        return f"hitl_crops/package/{key}"

    def _choose_annotation_package(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Open annotation package")
        if directory:
            try:
                self.open_annotation_package(Path(directory))
            except (OSError, ValueError, KeyError, TypeError) as error:
                QMessageBox.warning(self, "Annotation package", str(error))

    def open_annotation_package(self, root: Path) -> None:
        """Resume a complete local package and restore verified saved progress."""
        values = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if values.get("managed_annotation_version") is None:
            context = HitlCropContext.read(root / "roi_context.json")
            if self._store is not None:
                context.validate_store(self._store, self._platform)
            migrated = upgrade_legacy_annotation_package(
                root,
                self._store.parent if self._store is not None else root.parent,
                context,
            )
            root = migrated.root
            QMessageBox.information(
                self,
                "Existing masks preserved",
                "The older package was copied into a new annotation folder. "
                "Open and save each crop once to record its completion. "
                "Your original files and mask edits are unchanged.",
            )
        package = open_annotation_package(root)
        context = HitlCropContext(**package.manifest["roi_context"])
        if self._store is not None:
            context.validate_store(self._store, self._platform)
        rectangles = []
        for crop in package.crops:
            y0, y1, x0, x1 = crop.bounds
            rectangles.append(np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], float))
        session = _CropSession(context, rectangles, root / "roi_context.json", package)
        if self._store is None:
            self._offline_session = session
        else:
            self._sessions[str(self._store)] = session
            self._settings.setValue(self._package_setting_key(), str(root))
        self._restore_layers()

    def _prepare_annotation(self) -> None:
        session = self._session
        if (
            session is None
            or session.setup_path is None
            or self._export_worker is not None
        ):
            return
        try:
            self._require_native_image()
            validate_crop_rectangles(session.context, session.rectangles)
            settings = load_annotation_export_settings(session.setup_path)
            density_path = None
            if "transcript_density" in settings.channels:
                density_path = find_local_density(
                    self._store, session.setup_path, settings
                )
                remembered = self._settings.value(
                    "hitl_crops/last_density", "", type=str
                )
                if density_path is None and remembered:
                    density_path = Path(remembered)
                if density_path is not None:
                    try:
                        validate_local_density(density_path, session.context, settings)
                    except (OSError, ValueError, KeyError, TypeError):
                        density_path = None
                if density_path is None:
                    chosen = QFileDialog.getExistingDirectory(
                        self,
                        "Choose the matching transcript_density.zarr",
                        str(self._store.parent),
                    )
                    if not chosen:
                        return
                    density_path = Path(chosen)
                    validate_local_density(density_path, session.context, settings)
                self._settings.setValue("hitl_crops/last_density", str(density_path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.warning(self, "Prepare annotation", str(error))
            return
        source, context = self._store, session.context
        rectangles = [points.copy() for points in session.rectangles]
        self._save_button.setEnabled(False)
        self._add_button.setEnabled(False)
        self._remove_button.setEnabled(False)
        self._layer.editable = False
        self._status.setText(
            "Preparing Cellpose images, prefills and training inputs "
            "beside the dataset…"
        )

        def compute():
            return export_local_annotation_package(
                source, context, rectangles, settings, density_path=density_path
            )

        worker = thread_worker(compute)()
        worker.returned.connect(
            lambda package: self._annotation_prepared(source, package)
        )
        worker.errored.connect(
            lambda error: self._annotation_export_failed(source, error)
        )
        worker.finished.connect(lambda: setattr(self, "_export_worker", None))
        self._export_worker = worker
        worker.start()

    def _annotation_prepared(self, source: Path, package: AnnotationPackage) -> None:
        session = self._sessions.get(str(source))
        if session is not None:
            session.package = package
        key = hashlib.sha256(str(source).encode()).hexdigest()
        self._settings.setValue(f"hitl_crops/package/{key}", str(package.root))
        if self._store == source:
            self.open_annotation_package(package.root)
            self._status.setText(
                "Package ready. Select a crop in the list and press "
                "Annotate cell masks."
            )

    def _annotation_export_failed(self, source: Path, error: Exception) -> None:
        if self._store == source:
            if self._layer is not None:
                self._layer.editable = True
            self._add_button.setEnabled(True)
            self._sync_selection()
            self._status.setText(f"Package preparation failed: {error}")
        QMessageBox.warning(self, "Prepare annotation", str(error))

    def _new_crop_selection(self) -> None:
        if self._session is None or self._export_worker is not None:
            return
        if self._store is None:
            self._offline_session = None
            self.set_dataset(None)
            return
        session = self._session
        session.package = None
        session.rectangles = []
        self._settings.remove(self._package_setting_key())
        self._restore_layers()
        self._annotate_button.setEnabled(False)

    def _choose_cellpose_python(self) -> str:
        executable, _ = QFileDialog.getOpenFileName(
            self, "Choose Python from the Cellpose 4.2.1.1 drawing environment"
        )
        if executable:
            self._settings.setValue("hitl_crops/cellpose_python", executable)
        return executable

    def _annotate_selected_crop(self) -> None:
        session = self._session
        row = self._crop_list.currentRow()
        if (
            session is None
            or session.package is None
            or row < 0
            or self._annotation_launch is not None
        ):
            return
        executable = self._settings.value("hitl_crops/cellpose_python", "", type=str)
        if not executable or not Path(executable).is_file():
            executable = self._choose_cellpose_python()
        if not executable:
            return
        package = session.package
        crop = package.crops[row]
        nonce = uuid.uuid4().hex
        try:
            # Check immutable inputs again immediately before launching an editor.
            package = open_annotation_package(package.root)
            runtime = package.root / "_runtime"
            if not runtime.resolve().is_relative_to(package.root):
                raise ValueError("Annotation runtime must stay inside its package.")
            runtime.mkdir(exist_ok=True)
            runner = runtime / "cellpose_annotation_runner.py"
            if not runner.resolve().is_relative_to(package.root):
                raise ValueError("Annotation launcher must stay inside its package.")
            runner.write_text(
                files("napari_compare_xenium_merscope")
                .joinpath("resources/cellpose_annotation_runner.txt")
                .read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            log_path = runtime / f"{crop.crop_id}-{nonce}.log"
            environment = dict(os.environ)
            for key in (
                "PYTHONHOME",
                "PYTHONPATH",
                "PYTHONEXECUTABLE",
                "QT_PLUGIN_PATH",
                "QT_QPA_PLATFORM_PLUGIN_PATH",
                "QT_API",
            ):
                environment.pop(key, None)
            options = (
                {"start_new_session": True}
                if os.name != "nt"
                else {"creationflags": subprocess.CREATE_NO_WINDOW}
            )
            with log_path.open("wb") as log:
                process = subprocess.Popen(
                    [
                        executable,
                        "-I",
                        "-u",
                        str(runner),
                        "--package",
                        str(package.root),
                        "--crop",
                        crop.crop_id,
                        "--session",
                        nonce,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    **options,
                )
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "Launch Cellpose", str(error))
            return
        self._annotation_launch = (
            package,
            row,
            nonce,
            process,
            time.monotonic(),
            log_path,
        )
        self._annotate_button.setEnabled(False)
        self._status.setText(
            "Edit this crop in Cellpose, press Ctrl+S, then close Cellpose to continue."
        )
        self._gui_timer.start()

    def _poll_annotation_gui(self) -> None:
        launch = self._annotation_launch
        if launch is None or launch[3].poll() is None:
            return
        self._gui_timer.stop()
        self._annotation_launch = None
        package, row, nonce, process, started, log_path = launch
        try:
            validate_saved_annotation(package, package.crops[row], session_nonce=nonce)
        except (OSError, ValueError, KeyError, TypeError):
            if process.returncode:
                message = (
                    "Cellpose did not finish normally. Choose a Cellpose 4.2.1.1 "
                    f"drawing environment and try again. Details: {log_path}"
                )
            else:
                message = (
                    "No new verified save was found. Reopen this crop, "
                    "press Ctrl+S, then close Cellpose."
                )
            if (
                self._session is not None
                and self._session.package is not None
                and self._session.package.root == package.root
            ):
                self._update_annotation_state()
                self._status.setText(message)
            QMessageBox.warning(self, "Crop still needs annotation", message)
            return
        self._synchronize_review(
            package, (package.crops[row].crop_id, (time.monotonic() - started) / 60)
        )
        if (
            self._session is not None
            and self._session.package is not None
            and self._session.package.root == package.root
        ):
            self._update_annotation_state(prompt=True)

    def _synchronize_review(self, package: AnnotationPackage, timed_crop=None) -> None:
        completed = annotation_progress(package)
        review = package.root / "review.tsv"
        with review.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            rows = list(reader)
            fields = reader.fieldnames or []
        if not {"crop_id", "status", "reason", "minutes", "notes"}.issubset(fields):
            raise ValueError("The review sheet is missing annotation columns.")
        if {row["crop_id"] for row in rows} != {crop.crop_id for crop in package.crops}:
            raise ValueError("Review sheet no longer matches the annotation package.")
        for row in rows:
            if row["crop_id"] in completed and row.get("status") != "excluded":
                row["status"] = "reviewed"
            elif row.get("status") == "reviewed":
                row["status"] = "pending"
            if timed_crop is not None and row["crop_id"] == timed_crop[0]:
                try:
                    previous = float(row.get("minutes") or 0)
                except ValueError:
                    previous = 0
                row["minutes"] = f"{previous + timed_crop[1]:.1f}"
        if not review.resolve().is_relative_to(package.root):
            raise ValueError("The review sheet must stay inside its package.")
        temporary = review.with_name(f".review-{uuid.uuid4().hex}.tsv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fields,
                delimiter="\t",
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, review)

    def _update_annotation_state(self, *, prompt: bool = False) -> None:
        package = self._session.package
        if package is None:
            return
        self._synchronize_review(package)
        completed = annotation_progress(package)
        self._render_list()
        self._new_selection_button.setEnabled(True)
        self._add_button.setEnabled(False)
        self._remove_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._regions_button.setEnabled(False)
        self._annotate_button.setEnabled(
            self._crop_list.currentRow() >= 0 and self._annotation_launch is None
        )
        cells = sum(item.cell_count for item in completed.values())
        self._counter.setText(
            f"Annotated crops: {len(completed)}/{len(package.crops)} · "
            f"{cells:,} saved cells"
        )
        self._target.setText("Annotate every selected crop")
        self._target.setStyleSheet("")
        self._target.setProperty("targetReached", False)
        if len(completed) == len(package.crops):
            message = (
                f"Annotation complete. Output folder:\n{package.root}\n\n"
                f"Server relative folder:\n{package.transfer_relative_path}\n"
                "Place it under the configured MERXEN_HITL_ROOT.\n\n"
                "In MerXen, start at train_cellpose and supply this package ID "
                "and a unique custom model name."
            )
            self._status.setText(message)
            if prompt:
                QMessageBox.information(self, "Annotation complete", message)
        else:
            pending = next(
                index
                for index, crop in enumerate(package.crops)
                if crop.crop_id not in completed
            )
            self._crop_list.setCurrentRow(pending)
            self._select_crop(pending)
            self._status.setText(
                f"Select crop {pending + 1} and press Annotate cell masks. "
                "Save in Cellpose, then close it to proceed."
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
