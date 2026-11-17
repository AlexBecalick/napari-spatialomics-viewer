"""Tests of the manual-registration controller on ``napari.components.ViewerModel`` (offscreen)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import manual_registration_stores as stores
import numpy as np
import pytest
from napari.components import ViewerModel
from napari.layers import Points
from napari.utils.key_bindings import KeyBinding

from napari_compare_xenium_merscope import manual_registration_controller as controller_module
from napari_compare_xenium_merscope.manual_registration_controller import (
    CROP_MIN_DRAW_PX,
    DISABLED_PAIR_COLOUR,
    ManualRegistrationController,
    RegistrationLoadCancelled,
    contrast_limits_from_level,
    contrast_limits_from_tissue_tiles,
    crop_from_drag,
    crop_rectangle_rc,
    label_translation_data,
    marker_data_size,
    marker_world_size,
    point_inside_image,
    result_banner,
    result_rows,
    snapped_crop_size,
)
from napari_compare_xenium_merscope.manual_registration_model import (
    LABELLING_ATTR,
    LAYER_METADATA_KEY,
    STORE_ATTR,
    CoarseState,
    CropRect,
    coarse_matrix_xy,
    evaluate_landmarks,
    finish_allowed,
    labelling_from_attrs,
    napari_affine_rc,
    pair_colour,
    parse_store_descriptor,
)
from napari_compare_xenium_merscope.manual_registration_store import (
    StoreBusyError,
    read_local_draft,
    read_root_attrs,
    save_labelling,
    write_local_draft,
)
from napari_compare_xenium_merscope.manual_registration_widget import ManualRegistrationWidget

REF_SHAPE = stores.REFERENCE_SHAPE_YX
MOV_SHAPE = stores.MOVING_SHAPE_YX
P_R = stores.REFERENCE_PIXEL_SIZE_UM
P_M = stores.MOVING_PIXEL_SIZE_UM
GOLDEN = Path(__file__).parent / "data" / "manual_registration_golden_v1.json"


# ---------------------------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("qtpy/Qt not available")
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="session")
def prepared_template(tmp_path_factory):
    return stores.build_prepared_store(tmp_path_factory.mktemp("prepared_tpl") / "tpl.zarr")


@pytest.fixture(scope="session")
def result_template(tmp_path_factory):
    return stores.build_result_store(tmp_path_factory.mktemp("result_tpl") / "tpl.zarr")


@pytest.fixture
def prepared_store(prepared_template, tmp_path):
    target = tmp_path / "SYN1_prepared.zarr"
    shutil.copytree(prepared_template, target)
    return target


@pytest.fixture
def result_store(result_template, tmp_path):
    target = tmp_path / "SYN1_registered.zarr"
    shutil.copytree(result_template, target)
    return target


@pytest.fixture
def viewer(qapp):
    return ViewerModel()


@pytest.fixture
def host(viewer, tmp_path):
    return stores.FakeHost(viewer, tmp_path / "appdata")


@pytest.fixture
def widget(qapp):
    return ManualRegistrationWidget()


@pytest.fixture
def ctrl(host, widget, monkeypatch):
    controller = ManualRegistrationController(host)
    controller.set_widget(widget)
    dialogs = SimpleNamespace(asked=[], warned=[], answer="Cancel")

    def ask(title, text, buttons, *, default):
        dialogs.asked.append((title, text, tuple(buttons)))
        return dialogs.answer if dialogs.answer in buttons else default

    monkeypatch.setattr(controller, "_ask", ask)
    monkeypatch.setattr(
        controller, "_warn", lambda title, text: dialogs.warned.append((title, text))
    )
    controller.dialogs = dialogs
    return controller


@pytest.fixture
def loaded(ctrl, prepared_store):
    assert ctrl.load_store(prepared_store)
    ctrl.flush_pending_updates()
    return ctrl


def make_env(app_dir, *, deferred=False, answer="Cancel"):
    """Build a second viewer/host/widget/controller (for reload tests)."""
    viewer = ViewerModel()
    host = (stores.DeferredHost if deferred else stores.FakeHost)(viewer, app_dir)
    widget = ManualRegistrationWidget()
    controller = ManualRegistrationController(host)
    controller.set_widget(widget)
    dialogs = SimpleNamespace(asked=[], warned=[], answer=answer)

    def ask(title, text, buttons, *, default):
        dialogs.asked.append((title, text, tuple(buttons)))
        return dialogs.answer if dialogs.answer in buttons else default

    controller._ask = ask
    controller._warn = lambda title, text: dialogs.warned.append((title, text))
    controller.dialogs = dialogs
    return viewer, host, widget, controller


def to_hex(rgba):
    return "#" + "".join(f"{round(255 * float(c)):02x}" for c in rgba[:3])


def gesture(ctrl, world_rc, *, end_rc=None, drag_px=0.0, button=1):
    ev = SimpleNamespace(
        type="mouse_press", button=button, position=tuple(world_rc), pos=np.array([100.0, 100.0])
    )
    gen = ctrl.mouse_gesture(ev)
    if gen is None:
        return False
    next(gen)
    if drag_px:
        ev.type = "mouse_move"
        ev.pos = np.array([100.0 + drag_px, 100.0])
        if end_rc is not None:
            ev.position = tuple(end_rc)
        next(gen)
    ev.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(gen)
    return True


def world_of(layer, x, y):
    return layer.data_to_world((y, x))


def press_key(viewer, key):
    return viewer.keymap[KeyBinding.from_str(key)](viewer)


def click_pair(ctrl, viewer, ref_xy, mov_xy):
    if not ctrl._landmark_mode:
        ctrl._widget.landmark_mode_changed.emit(True)
    assert gesture(ctrl, world_of(viewer.layers["Reference | DAPI"], *ref_xy))
    assert gesture(ctrl, world_of(viewer.layers["Moving | DAPI"], *mov_xy))


TRUTH = CoarseState(12.0, False, (5.0, -2.5))
MOVING_POINTS = [(200, 150), (900, 200), (850, 800), (250, 750), (550, 480)]


def truth_matrix():
    return coarse_matrix_xy(
        TRUTH,
        moving_shape_yx=MOV_SHAPE,
        reference_shape_yx=REF_SHAPE,
        moving_pixel_size_um=P_M,
        reference_pixel_size_um=P_R,
    )


def truth_reference(mov_xy):
    m = truth_matrix()
    p = m @ np.array([mov_xy[0], mov_xy[1], 1.0])
    return (round(float(p[0]), 3), round(float(p[1]), 3))


def place_truth_pairs(ctrl, viewer, points=MOVING_POINTS):
    for mov in points:
        click_pair(ctrl, viewer, truth_reference(mov), mov)


def coarse_affine(coarse):
    return napari_affine_rc(
        np.diag([P_R, P_R, 1.0]),
        coarse_matrix_xy(
            coarse,
            moving_shape_yx=MOV_SHAPE,
            reference_shape_yx=REF_SHAPE,
            moving_pixel_size_um=P_M,
            reference_pixel_size_um=P_R,
        ),
    )


def layer_affine(layer):
    return np.asarray(layer.affine.affine_matrix)


def on_disk_labelling(path):
    return read_root_attrs(path).get(LABELLING_ATTR)


def grid_pairs(n):
    """``n`` well spread landmark pairs ``(reference_xy, moving_xy)``."""
    out = []
    for i in range(n):
        ref = (100.0 + 100.0 * (i % 4) + 7.0 * i, 150.0 + 200.0 * (i // 4) + 13.0 * i)
        mov = (120.0 + 90.0 * (i % 4) + 3.0 * i, 130.0 + 180.0 * (i // 4) + 9.0 * i)
        out.append((ref, mov))
    return out


def moving_world_centre(viewer):
    layer = viewer.layers["Moving | DAPI"]
    return np.asarray(world_of(layer, 549.5, 479.5))


# ---------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------


def test_marker_geometry_worked_example():
    assert marker_world_size(2.0) == pytest.approx(15.0)
    assert marker_world_size(0.0) is None and marker_world_size(float("nan")) is None
    ref = np.diag([0.5, 0.5, 1.0])
    assert marker_data_size(2.0, ref) == pytest.approx(30.0)
    assert marker_data_size(2.0, ref) * 0.6 == pytest.approx(18.0)
    assert label_translation_data(ref, 15.0) == pytest.approx([-13.5, 13.5])
    mov = np.diag([0.4, 0.4, 1.0])
    assert marker_data_size(2.0, mov) == pytest.approx(37.5)
    assert label_translation_data(mov, 15.0) == pytest.approx([-16.875, 16.875])


def test_crop_helpers():
    crop = crop_from_drag((100.2, 50.7), (600.6, 700.1), (1100, 1000))
    assert crop == CropRect(x0=100, y0=51, x1=602, y1=701)
    assert crop_from_drag((10.0, 10.0), (20.0, 20.0), (1100, 1000)) is None
    assert crop_from_drag((-50.0, -50.0), (2000.0, 3000.0), (1100, 1000)) == CropRect(
        0, 0, 1000, 1100
    )
    assert snapped_crop_size(crop, (1100, 1000)) == (448, 576)
    assert snapped_crop_size(None, (1100, 1000)) == (1000, 1100)
    assert snapped_crop_size(CropRect(0, 0, 1000, 1100), (1100, 1000)) == (1000, 1100)
    assert snapped_crop_size(CropRect(1, 1, 999, 1099), (1100, 1000)) == (896, 1024)
    assert crop_rectangle_rc(crop).tolist() == [
        [50.5, 99.5],
        [50.5, 601.5],
        [700.5, 601.5],
        [700.5, 99.5],
    ]
    assert point_inside_image((-0.5, -0.5), (1100, 1000))
    assert not point_inside_image((999.5, 0.0), (1100, 1000))
    assert not point_inside_image((0.0, -0.51), (1100, 1000))


def test_contrast_from_coarsest_level_pure():
    import xarray as xr

    zeros = xr.DataArray(np.zeros((1, 40, 40), dtype=np.uint16), dims=("c", "y", "x"))
    assert contrast_limits_from_level(zeros, 0) == (0.0, 1.0)
    constant = xr.DataArray(np.full((1, 40, 40), 7, dtype=np.uint16), dims=("c", "y", "x"))
    assert contrast_limits_from_level(constant, 0) == (7.0, 8.0)

    seen = []

    class Plane:
        shape = (3000, 3000)

        def __getitem__(self, key):
            seen.append(key)
            return np.ones((10, 10), dtype=np.uint16)

    level = SimpleNamespace(isel=lambda c: SimpleNamespace(shape=(3000, 3000), data=Plane()))
    contrast_limits_from_level(level, 0, max_pixels=1_000_000)
    assert seen == [(slice(None, None, 3), slice(None, None, 3))]


def test_result_rows_and_banner():
    records = stores.default_result_landmarks()
    records[0]["residual_um"].pop("affine")
    records[1]["residual_um"]["loo"] = None
    records.reverse()
    result = stores.result_attr(landmarks=records)
    rows = result_rows(result)
    assert [r.id for r in rows] == [1, 2, 3]
    assert rows[0].affine_um is None and rows[0].final_um == pytest.approx(0.5)
    assert np.isnan(rows[1].loo_um) and rows[1].fit_um == pytest.approx(3.0)
    assert rows[2].used is False and rows[2].outlier is True
    banner = result_banner(result)
    assert banner.splitlines() == [
        "SYN1_serial_section · preset serial_section",
        "Refinement: applied · Flow: skipped – too few confident windows",
        "Tissue Pearson: coarse 0.31 → landmark 0.52 → affine 0.58 → final 0.66",
    ]
    result["run_label"] = "trial"
    result["tissue_qc"] = {}
    lines = result_banner(result).splitlines()
    assert lines[0].endswith("preset serial_section (trial)")
    assert lines[2] == "Tissue Pearson: coarse n/a → landmark n/a → affine n/a → final n/a"


def test_module_has_no_second_fit_implementation():
    assert not hasattr(controller_module, "compute_live_fit")
    assert not hasattr(controller_module, "LiveFit")


# ---------------------------------------------------------------------------------------------
# 1-5 Loading, layers, affines
# ---------------------------------------------------------------------------------------------


def test_load_prepared_creates_tagged_layers(ctrl, viewer, host, prepared_store):
    assert ctrl.load_store(prepared_store)
    ctrl.flush_pending_updates()
    assert [layer.name for layer in viewer.layers] == [
        "Reference | DAPI",
        "Reference | AF568",
        "Moving | DAPI",
        "Moving | AF568",
        "Moving | AF647",
        "Reference crop",
        "Moving crop",
        "Moving landmarks",
        "Reference landmarks",
        "Suggested landmarks",
    ]
    expected_meta = {
        "Reference | DAPI": ("reference", "image", "DAPI"),
        "Reference | AF568": ("reference", "image", "AF568"),
        "Moving | DAPI": ("moving", "image", "DAPI"),
        "Moving | AF568": ("moving", "image", "AF568"),
        "Moving | AF647": ("moving", "image", "AF647"),
        "Reference crop": ("reference", "crop", None),
        "Moving crop": ("moving", "crop", None),
        "Moving landmarks": ("moving", "landmarks", None),
        "Reference landmarks": ("reference", "landmarks", None),
        "Suggested landmarks": ("reference", "overlay", None),
    }
    for layer in viewer.layers:
        role, kind, channel = expected_meta[layer.name]
        assert layer.metadata[LAYER_METADATA_KEY] == {
            "role": role,
            "kind": kind,
            "channel": channel,
        }
    for name in ("Reference | DAPI", "Moving | DAPI"):
        assert viewer.layers[name].visible
    for name in ("Reference | AF568", "Moving | AF568", "Moving | AF647"):
        assert not viewer.layers[name].visible
        assert viewer.layers[name].colormap.name == "gray"
    assert viewer.layers["Reference | DAPI"].colormap.name == "green"
    assert viewer.layers["Moving | DAPI"].colormap.name == "magenta"
    for layer in viewer.layers:
        if layer.metadata[LAYER_METADATA_KEY]["kind"] == "image":
            assert layer.blending == "additive"
            assert layer.multiscale and len(layer.data) == 2
    assert host.worker_keys == ["registration_load"]
    assert host.entered == [prepared_store]
    assert host.ready == [(prepared_store, "prepared")]
    assert viewer.grid.enabled is False
    assert viewer.layers.selection.active is viewer.layers["Reference | DAPI"]
    assert ctrl.has_unsaved_changes() is False
    assert ctrl.is_active() and ctrl.display_mode == "both"


def test_contrast_from_full_resolution_tissue_tiles(loaded, viewer):
    session = loaded.session
    for role, names in (
        ("reference", stores.REFERENCE_CHANNELS),
        ("moving", stores.MOVING_CHANNELS),
    ):
        for index, channel in enumerate(names):
            prefix = "Reference" if role == "reference" else "Moving"
            levels = session.levels[role]
            expected = contrast_limits_from_tissue_tiles(levels[0], levels[-1], index)
            assert viewer.layers[f"{prefix} | {channel}"].contrast_limits == pytest.approx(
                expected, rel=1e-6
            )


def test_identity_coarse_affines(loaded, viewer):
    ref_affine = layer_affine(viewer.layers["Reference | DAPI"])
    assert ref_affine == pytest.approx(np.diag([0.5, 0.5, 1.0]), abs=1e-12)
    mov = layer_affine(viewer.layers["Moving | DAPI"])
    assert mov == pytest.approx(coarse_affine(CoarseState()), abs=1e-12)
    assert mov[:2, :2] == pytest.approx(np.diag([0.4, 0.4]), abs=1e-12)
    for layer in viewer.layers:
        assert tuple(layer.scale) == (1.0, 1.0)
        assert tuple(layer.translate) == (0.0, 0.0)
        assert np.allclose(layer.rotate, np.eye(2))
    assert moving_world_centre(viewer) == pytest.approx([274.75, 249.75], abs=1e-9)


def test_coarse_rotation_mirror_offset(loaded, viewer, widget):
    widget.rotation_set.emit(178.5)
    widget.flip_horizontal.emit()
    widget.offset_set.emit(12.5, -7.5)
    loaded.flush_pending_updates()
    assert loaded.state.coarse == CoarseState(-178.5, True, (12.5, -7.5))
    affine = layer_affine(viewer.layers["Moving | DAPI"])
    assert np.linalg.det(affine[:2, :2]) < 0
    assert moving_world_centre(viewer) == pytest.approx([274.75 - 7.5, 249.75 + 12.5], abs=1e-9)
    names = [
        "Moving | DAPI",
        "Moving | AF568",
        "Moving | AF647",
        "Moving landmarks",
        "Moving crop",
    ]
    for name in names:
        assert layer_affine(viewer.layers[name]) == pytest.approx(affine, abs=1e-12)
    assert loaded.has_unsaved_changes()


def test_affine_updates_coalesce(loaded, viewer, widget):
    before = layer_affine(viewer.layers["Moving | DAPI"]).copy()
    for _ in range(5):
        widget.rotate_by.emit(1.0)
    assert layer_affine(viewer.layers["Moving | DAPI"]) == pytest.approx(before)
    loaded.flush_pending_updates()
    assert layer_affine(viewer.layers["Moving | DAPI"]) == pytest.approx(
        coarse_affine(CoarseState(5.0)), abs=1e-12
    )


# ---------------------------------------------------------------------------------------------
# 6-11 Clicks
# ---------------------------------------------------------------------------------------------


def test_click_sequence_under_rotation_and_mirror(loaded, viewer, widget):
    widget.rotation_set.emit(178.5)
    widget.flip_horizontal.emit()
    widget.offset_set.emit(12.5, -7.5)
    loaded.flush_pending_updates()
    widget.landmark_mode_changed.emit(True)
    assert loaded.display_mode == "reference"
    ref_layer, mov_layer = viewer.layers["Reference | DAPI"], viewer.layers["Moving | DAPI"]
    assert gesture(loaded, world_of(ref_layer, 200.5, 300.25))
    assert loaded.pending_reference_xy == pytest.approx((200.5, 300.25), abs=1e-9)
    assert loaded.display_mode == "moving"
    assert all(layer.opacity == 0 for layer in viewer.layers if layer.name.startswith("Reference"))
    assert viewer.layers["Moving | DAPI"].opacity == 1
    assert gesture(loaded, world_of(mov_layer, 640.125, 410.75))
    state = loaded.state
    assert len(state.pairs) == 1
    pair = state.pairs[0]
    assert pair.id == 1 and pair.colour == pair_colour(1, state.colour_seed)
    assert pair.reference_xy == pytest.approx((200.5, 300.25), abs=1e-9)
    assert pair.moving_xy == pytest.approx((640.125, 410.75), abs=1e-9)
    assert loaded.pending_reference_xy is None
    assert loaded.display_mode == "reference"
    assert state.next_id == 2
    assert loaded.has_unsaved_changes()


def test_preview_aligns_landmarks_and_clicks_stay_exact(loaded, viewer, widget):
    place_truth_pairs(loaded, viewer)
    widget.preview_changed.emit(True)
    loaded.flush_pending_updates()
    expected = coarse_affine(TRUTH)
    actual = layer_affine(viewer.layers["Moving | DAPI"])
    # The reference points are rounded to 3 decimals, so the fit is exact only to ~1e-4 um.
    assert actual == pytest.approx(expected, abs=2e-3)
    ref_layer = viewer.layers["Reference landmarks"]
    mov_layer = viewer.layers["Moving landmarks"]
    for mov in MOVING_POINTS:
        ref = truth_reference(mov)
        got = mov_layer.data_to_world((mov[1], mov[0]))
        want = ref_layer.data_to_world((ref[1], ref[0]))
        assert np.allclose(got, want, atol=1e-3)
    # clicks stay exact under the previewed transform
    assert gesture(loaded, world_of(viewer.layers["Reference | DAPI"], 300.0, 400.0))
    assert gesture(loaded, world_of(viewer.layers["Moving | DAPI"], 100.25, 200.5))
    assert loaded.state.pairs[-1].moving_xy == pytest.approx((100.25, 200.5), abs=1e-9)
    before = loaded.state.coarse
    widget.rotate_by.emit(10.0)
    assert loaded.state.coarse == before


def test_preview_status_text(loaded, viewer, widget, host):
    place_truth_pairs(loaded, viewer)
    widget.preview_changed.emit(True)
    widget.rotate_by.emit(10.0)
    assert host.statuses[-1] == "Turn off Preview fit to change the coarse alignment."


def test_both_mode_targets_by_pending(loaded, viewer, widget):
    widget.landmark_mode_changed.emit(True)
    widget.display_mode_changed.emit("both")
    assert loaded.display_mode == "both"
    assert gesture(loaded, world_of(viewer.layers["Reference | DAPI"], 200.0, 300.0))
    assert loaded.pending_reference_xy == (200.0, 300.0)
    assert loaded.display_mode == "both"
    assert gesture(loaded, world_of(viewer.layers["Moving | DAPI"], 250.0, 310.0))
    assert loaded.pending_reference_xy is None
    assert loaded.state.pairs[0].moving_xy == (250.0, 310.0)
    assert loaded.display_mode == "both"


def test_click_outside_image_rejected(loaded, widget, host):
    widget.landmark_mode_changed.emit(True)
    assert gesture(loaded, (-50.0, -50.0))
    assert loaded.pending_reference_xy is None
    assert host.statuses[-1] == "Click inside the reference image."


def test_drag_is_not_a_click(loaded, viewer, widget):
    widget.landmark_mode_changed.emit(True)
    world = world_of(viewer.layers["Reference | DAPI"], 200.0, 300.0)
    assert gesture(
        loaded, world, end_rc=world_of(viewer.layers["Reference | DAPI"], 260.0, 330.0), drag_px=10
    )
    assert loaded.pending_reference_xy is None
    # a movement below the threshold still counts as a click
    assert gesture(loaded, world, drag_px=2)
    assert loaded.pending_reference_xy == (200.0, 300.0)


def test_gesture_not_consumed(loaded, viewer, widget, ctrl, result_store, tmp_path):
    ev = SimpleNamespace(
        type="mouse_press", button=1, position=(10.0, 10.0), pos=np.array([1.0, 1.0])
    )
    assert loaded.mouse_gesture(ev) is None  # landmark mode off
    widget.landmark_mode_changed.emit(True)
    assert gesture(loaded, (10.0, 10.0), button=2) is False  # right button
    ev.type = "mouse_move"
    assert loaded.mouse_gesture(ev) is None  # not a press
    _v, _h, _w, other = make_env(tmp_path / "app2")
    assert other.load_store(result_store)
    assert gesture(other, (10.0, 10.0)) is False


# ---------------------------------------------------------------------------------------------
# 12-18 Keys, undo, fit, markers
# ---------------------------------------------------------------------------------------------


def opacity(viewer, name):
    return viewer.layers[name].opacity


def test_flick_and_peek_keys(loaded, viewer, widget):
    widget.display_mode_changed.emit("reference")
    press_key(viewer, "F")
    assert loaded.display_mode == "moving"
    assert opacity(viewer, "Moving | DAPI") == 1 and opacity(viewer, "Reference | DAPI") == 0
    press_key(viewer, "F")
    assert loaded.display_mode == "reference"
    gen = press_key(viewer, "Shift-F")
    next(gen)
    assert loaded.display_mode == "reference"
    assert opacity(viewer, "Moving | DAPI") == 1 and opacity(viewer, "Reference | DAPI") == 0
    assert opacity(viewer, "Moving landmarks") == 1 and opacity(viewer, "Reference landmarks") == 0
    with pytest.raises(StopIteration):
        next(gen)
    assert opacity(viewer, "Reference | DAPI") == 1 and opacity(viewer, "Moving | DAPI") == 0
    assert viewer.layers["Reference | DAPI"].visible and viewer.layers["Moving | DAPI"].visible


def test_flick_respects_user_opacity(loaded, viewer, widget):
    viewer.layers["Moving | DAPI"].opacity = 0.4
    widget.display_mode_changed.emit("reference")
    assert opacity(viewer, "Moving | DAPI") == 0
    widget.display_mode_changed.emit("moving")
    assert opacity(viewer, "Moving | DAPI") == pytest.approx(0.4)


def test_ctrl_z_cancels_pending_first(loaded, viewer):
    click_pair(loaded, viewer, (200.0, 300.0), (210.0, 310.0))
    assert gesture(loaded, world_of(viewer.layers["Reference | DAPI"], 400.0, 500.0))
    assert loaded.pending_reference_xy is not None
    press_key(viewer, "Control-Z")
    assert loaded.pending_reference_xy is None and len(loaded.state.pairs) == 1
    press_key(viewer, "Control-Z")
    assert len(loaded.state.pairs) == 0
    click_pair(loaded, viewer, (220.0, 320.0), (230.0, 330.0))
    assert [p.id for p in loaded.state.pairs] == [2]


def test_delete_and_undo_restore_ids_and_colours(loaded, viewer, widget):
    for ref, mov in grid_pairs(3):
        click_pair(loaded, viewer, ref, mov)
    before = [(p.id, p.colour, p.reference_xy, p.moving_xy, p.enabled) for p in loaded.state.pairs]
    widget.delete_pairs.emit([1, 3])
    assert [p.id for p in loaded.state.pairs] == [2]
    widget.undo_requested.emit()
    after = [(p.id, p.colour, p.reference_xy, p.moving_xy, p.enabled) for p in loaded.state.pairs]
    assert after == before
    click_pair(loaded, viewer, (500.0, 500.0), (510.0, 510.0))
    assert loaded.state.pairs[-1].id == 4


def test_enable_toggle_held_out_and_undo(loaded, viewer, widget):
    place_truth_pairs(loaded, viewer, MOVING_POINTS[:4])
    rows = []
    original = widget.set_rows
    widget.set_rows = lambda r, **kw: (rows.append(r), original(r, **kw))
    summaries = []
    original_summary = widget.set_fit_summary
    widget.set_fit_summary = lambda s: (summaries.append(s), original_summary(s))
    widget.pair_enabled_changed.emit(2, False)
    row2 = next(r for r in rows[-1] if r.id == 2)
    assert row2.used is False and row2.enabled is False and row2.fit_um < 5e-3
    assert summaries[-1].n_enabled == 3
    widget.undo_requested.emit()
    assert loaded.state.pair(2).enabled is True
    assert next(r for r in rows[-1] if r.id == 2).used is True


def test_live_fit_uses_evaluate_landmarks(loaded, viewer, widget, monkeypatch, prepared_store):
    rows_seen, summaries = [], []
    original_rows, original_summary = widget.set_rows, widget.set_fit_summary
    widget.set_rows = lambda r, **kw: (rows_seen.append(r), original_rows(r, **kw))
    widget.set_fit_summary = lambda s: (summaries.append(s), original_summary(s))
    points = MOVING_POINTS[:3]

    click_pair(loaded, viewer, truth_reference(points[0]), points[0])
    assert np.isnan(rows_seen[-1][0].fit_um) and np.isnan(rows_seen[-1][0].loo_um)
    assert summaries[-1] is None
    click_pair(loaded, viewer, truth_reference(points[1]), points[1])
    assert all(r.fit_um < 5e-3 for r in rows_seen[-1])
    assert all(np.isnan(r.loo_um) for r in rows_seen[-1])
    click_pair(loaded, viewer, truth_reference(points[2]), points[2])
    assert all(np.isfinite(r.loo_um) for r in rows_seen[-1])

    expected = evaluate_landmarks(
        loaded.state, moving_pixel_size_um=P_M, reference_pixel_size_um=P_R, outlier_floor_um=10.0
    )
    live = loaded._live_fit
    assert len(live.rows) == len(expected.rows)
    for got, want in zip(live.rows, expected.rows, strict=True):
        assert (got.id, got.used, got.outlier, got.outside_crop) == (
            want.id,
            want.used,
            want.outlier,
            want.outside_crop,
        )
        assert np.array_equal([got.fit_um, got.loo_um], [want.fit_um, want.loo_um], equal_nan=True)

    calls = []
    real = controller_module.evaluate_landmarks

    def recording(state, **kwargs):
        calls.append(kwargs)
        return real(state, **kwargs)

    monkeypatch.setattr(controller_module, "evaluate_landmarks", recording)
    dirty_before = loaded.has_unsaved_changes()
    widget.outlier_floor_changed.emit(30.0)
    assert calls and calls[-1]["outlier_floor_um"] == 30.0
    assert loaded.has_unsaved_changes() == dirty_before
    widget.save_requested.emit()
    assert on_disk_labelling(prepared_store)["outlier_floor_um"] == 30.0


def test_marker_rows_labels_and_widths(loaded, viewer, widget):
    for ref, mov in grid_pairs(2):
        click_pair(loaded, viewer, ref, mov)
    assert loaded.state.pair(1).reference_xy[0] < 149.5 < loaded.state.pair(2).reference_xy[0]
    widget.crop_set.emit("reference", CropRect(0, 0, 150, 600))
    assert gesture(loaded, world_of(viewer.layers["Reference | DAPI"], 300.0, 400.0))  # pending
    layer = viewer.layers["Reference landmarks"]
    assert len(layer.data) == 5
    assert [str(s) for s in layer.symbol] == ["ring", "cross", "ring", "cross", "ring"]
    assert layer.features["label"].tolist() == ["1", "", "2*", "", "3"]
    assert layer.border_width == pytest.approx([0.16, 0.30, 0.16, 0.30, 0.16])
    assert np.all(np.asarray(layer.border_color)[:, 3] == 0)
    assert layer.features["colour"].tolist() == [to_hex(c) for c in layer.face_color]
    assert layer.features["colour"].tolist()[4] == pair_colour(3, loaded.state.colour_seed)
    assert len(viewer.layers["Moving landmarks"].data) == 4
    assert viewer.layers["Moving landmarks"].features["label"].tolist() == ["1", "", "2*", ""]
    widget.pair_enabled_changed.emit(1, False)
    assert layer.features["colour"].tolist()[0] == DISABLED_PAIR_COLOUR
    default_limits = Points(np.empty((0, 2))).canvas_size_limits
    assert tuple(layer.canvas_size_limits) == tuple(default_limits)
    assert tuple(viewer.layers["Moving landmarks"].canvas_size_limits) == tuple(default_limits)


def test_marker_size_and_label_follow_zoom(loaded, viewer, widget):
    click_pair(loaded, viewer, (200.0, 300.0), (210.0, 310.0))
    viewer.camera.zoom = 2.0
    ref = viewer.layers["Reference landmarks"]
    mov = viewer.layers["Moving landmarks"]
    assert np.asarray(ref.size) == pytest.approx([30.0, 18.0])
    assert np.asarray(ref.text.translation) == pytest.approx([-13.5, 13.5])
    assert np.asarray(mov.size) == pytest.approx([37.5, 22.5])
    assert np.asarray(mov.text.translation) == pytest.approx([-16.875, 16.875])
    widget.rotation_set.emit(180.0)
    loaded.flush_pending_updates()
    linear = layer_affine(mov)[:2, :2]
    assert np.asarray(mov.text.translation) == pytest.approx(
        np.linalg.solve(linear, [-6.75, 6.75]), abs=1e-9
    )
    assert np.asarray(mov.text.translation) == pytest.approx([16.875, -16.875], abs=1e-9)
    assert np.asarray(mov.size) == pytest.approx([37.5, 22.5])


# ---------------------------------------------------------------------------------------------
# 19-21 Crops and guards
# ---------------------------------------------------------------------------------------------


def test_crop_draw_by_drag_and_undo(loaded, viewer, widget, host):
    restore_before = [f.__name__ for f in viewer.mouse_double_click_callbacks]
    widget.crop_draw_requested.emit("reference")
    assert loaded.display_mode == "reference"
    assert "double_click_to_zoom" not in [f.__name__ for f in viewer.mouse_double_click_callbacks]
    layer = viewer.layers["Reference | DAPI"]
    assert gesture(
        loaded,
        world_of(layer, 100.2, 50.7),
        end_rc=world_of(layer, 600.6, 700.1),
        drag_px=50,
    )
    assert loaded.state.crops["reference"] == CropRect(100, 51, 602, 701)
    crop_layer = viewer.layers["Reference crop"]
    assert len(crop_layer.data) == 1 and crop_layer.shape_type == ["rectangle"]
    assert np.asarray(crop_layer.data[0]) == pytest.approx(
        crop_rectangle_rc(CropRect(100, 51, 602, 701))
    )
    assert loaded.display_mode == "both"
    assert [f.__name__ for f in viewer.mouse_double_click_callbacks] == restore_before
    assert "448×576" in host.statuses[-1] and "1024" in host.statuses[-1]
    assert loaded.has_unsaved_changes()
    widget.undo_requested.emit()
    assert loaded.state.crops["reference"] is None
    assert len(viewer.layers["Reference crop"].data) == 0
    assert not loaded.has_unsaved_changes()


def test_moving_crop_under_rotated_mirrored_coarse(loaded, viewer, widget, host):
    widget.rotation_set.emit(178.5)
    widget.flip_horizontal.emit()
    widget.offset_set.emit(12.5, -7.5)
    loaded.flush_pending_updates()
    widget.crop_draw_requested.emit("moving")
    layer = viewer.layers["Moving | DAPI"]
    # corners given in MOVING pixels; the layer's affine puts them at the rotated world positions
    start_xy, end_xy = (100.2, 50.7), (600.6, 700.1)
    assert gesture(
        loaded,
        world_of(layer, *start_xy),
        end_rc=world_of(layer, *end_xy),
        drag_px=50,
    )
    assert loaded.state.crops["moving"] == CropRect(100, 51, 602, 701)  # moving pixels
    crop_layer = viewer.layers["Moving crop"]
    assert layer_affine(crop_layer) == pytest.approx(layer_affine(layer), abs=1e-12)
    assert np.asarray(crop_layer.data[0]) == pytest.approx(
        crop_rectangle_rc(CropRect(100, 51, 602, 701))
    )
    # the drawn rectangle's world corners sit on the image corners they were dragged between
    world = np.array([crop_layer.data_to_world(v) for v in np.asarray(crop_layer.data[0])])
    expected = np.array([layer.data_to_world((y, x)) for x, y in (start_xy, end_xy)])
    for corner in expected:  # within the integer snapping of the corners (0.2 px here)
        assert np.min(np.linalg.norm(world - corner, axis=1)) < 0.5
    assert loaded.display_mode == "both"


def test_crop_draw_too_small_stays_armed(loaded, viewer, widget, host):
    widget.crop_draw_requested.emit("moving")
    layer = viewer.layers["Moving | DAPI"]
    assert gesture(
        loaded, world_of(layer, 100.0, 100.0), end_rc=world_of(layer, 105.0, 105.0), drag_px=30
    )
    assert loaded.state.crops["moving"] is None
    assert (
        host.statuses[-1]
        == f"Crop too small: drag at least {CROP_MIN_DRAW_PX} px in the moving image."
    )
    assert loaded.display_mode == "moving"
    press_key(viewer, "Escape")
    assert loaded.display_mode == "both"


def test_crop_spinbox_validation(loaded, widget, host):
    widget.crop_set.emit("moving", CropRect(0, 0, 1200, 500))
    assert loaded.state.crops["moving"] is None
    assert host.statuses[-1].startswith("Crop must lie inside the moving image")
    widget.crop_set.emit("moving", CropRect(10, 20, 1100, 960))
    assert loaded.state.crops["moving"] == CropRect(10, 20, 1100, 960)
    widget.undo_requested.emit()
    assert loaded.state.crops["moving"] is None
    widget.crop_set.emit("moving", CropRect(0, 0, 8, 500))
    assert loaded.state.crops["moving"] is None


def test_guards_double_click_transform_grid(loaded, viewer, widget, host):
    names = lambda: [f.__name__ for f in viewer.mouse_double_click_callbacks]  # noqa: E731
    assert "double_click_to_zoom" in names()
    widget.landmark_mode_changed.emit(True)
    assert "double_click_to_zoom" not in names()
    widget.landmark_mode_changed.emit(False)
    assert "double_click_to_zoom" in names()
    assert names().count("double_click_to_zoom") == 1
    layer = viewer.layers["Moving | DAPI"]
    layer.mode = "transform"
    assert str(layer.mode) == "pan_zoom"
    assert host.statuses[-1].startswith("Transform mode is disabled")
    n_layers = len(viewer.layers)
    press_key(viewer, "Control-Alt-T")
    assert host.statuses[-1].startswith("Transform mode is disabled")
    press_key(viewer, "Control-Backspace")
    assert host.statuses[-1].startswith("Layers of a registration session cannot be deleted")
    press_key(viewer, "Control-Delete")
    assert len(viewer.layers) == n_layers
    assert viewer.grid.enabled is False


def test_active_layer_never_a_marker(loaded, viewer):
    viewer.layers.selection.active = viewer.layers["Reference landmarks"]
    assert viewer.layers.selection.active is viewer.layers["Reference | DAPI"]
    viewer.layers.selection.active = viewer.layers["Moving crop"]
    assert viewer.layers.selection.active is viewer.layers["Reference | DAPI"]


# ---------------------------------------------------------------------------------------------
# 22-27 Saving, drafts, conflicts
# ---------------------------------------------------------------------------------------------


def test_save_and_reload_restore_everything(loaded, viewer, widget, host, prepared_store, tmp_path):
    widget.flip_horizontal.emit()
    widget.rotation_set.emit(30.0)
    widget.offset_set.emit(12.5, -40.0)
    loaded.flush_pending_updates()
    widget.crop_set.emit("reference", CropRect(64, 64, 960, 1024))
    for ref, mov in grid_pairs(3):
        click_pair(loaded, viewer, ref, mov)
    widget.pair_enabled_changed.emit(2, False)
    assert loaded.has_unsaved_changes()
    widget.save_requested.emit()
    assert host.worker_keys[-1] == "registration_save"
    saved = on_disk_labelling(prepared_store)
    assert saved["revision"] == 1 and saved["status"] == "draft"
    assert STORE_ATTR in read_root_attrs(prepared_store)
    assert loaded.has_unsaved_changes() is False
    assert loaded.state.revision == 1 and loaded.state.status == "draft"
    assert read_local_draft(prepared_store, host.app_data_dir()) is None

    original = loaded.state
    v2, _h2, _w2, other = make_env(tmp_path / "appdata")
    assert other.load_store(prepared_store)
    restored = other.state
    assert restored.coarse == CoarseState(30.0, True, (12.5, -40.0))
    assert restored.crops == original.crops
    assert [(p.id, p.colour, p.enabled, p.reference_xy, p.moving_xy) for p in restored.pairs] == [
        (p.id, p.colour, p.enabled, p.reference_xy, p.moving_xy) for p in original.pairs
    ]
    assert restored.next_id == original.next_id
    assert restored.colour_seed == original.colour_seed
    assert restored.status == "draft" and restored.revision == 1
    assert other.has_unsaved_changes() is False
    for name in ("Reference landmarks", "Moving landmarks"):
        assert np.array_equal(v2.layers[name].data, viewer.layers[name].data)
        assert (
            v2.layers[name].features["label"].tolist()
            == viewer.layers[name].features["label"].tolist()
        )
    assert layer_affine(v2.layers["Moving | DAPI"]) == pytest.approx(
        layer_affine(viewer.layers["Moving | DAPI"]), abs=1e-12
    )
    assert len(v2.layers["Reference crop"].data) == 1


def test_finish_requires_eight_enabled_pairs(loaded, viewer, widget, host, prepared_store):
    pairs = grid_pairs(8)
    for ref, mov in pairs[:7]:
        click_pair(loaded, viewer, ref, mov)
    widget.finish_requested.emit()
    assert on_disk_labelling(prepared_store) is None
    assert host.statuses[-1] == finish_allowed(7, False, False)[1]
    click_pair(loaded, viewer, *pairs[7])
    widget.finish_requested.emit()
    written = on_disk_labelling(prepared_store)
    assert written["status"] == "finished" and written["revision"] == 1
    assert "Labelling finished: 8 enabled pairs saved (revision 1)" in host.statuses[-1]
    assert loaded.state.status == "finished"


def test_finish_blocked_by_pending_point(loaded, viewer, widget, host, prepared_store):
    for ref, mov in grid_pairs(8):
        click_pair(loaded, viewer, ref, mov)
    assert gesture(loaded, world_of(viewer.layers["Reference | DAPI"], 300.0, 400.0))
    widget.finish_requested.emit()
    assert on_disk_labelling(prepared_store) is None
    assert host.statuses[-1] == finish_allowed(8, True, False)[1]


def test_stale_revision_conflict(loaded, viewer, widget, host, prepared_store):
    click_pair(loaded, viewer, *grid_pairs(1)[0])
    widget.save_requested.emit()
    assert loaded.state.revision == 1
    other_pair = {"id": 1, "reference_xy_px": [700.0, 700.0], "moving_xy_px": [650.0, 640.0]}
    other = stores.labelling_payload(pairs=[other_pair, {**other_pair, "id": 2}], revision=1)
    save_labelling(prepared_store, other, expected_revision=1, software_version="test")
    assert on_disk_labelling(prepared_store)["revision"] == 2

    click_pair(loaded, viewer, *grid_pairs(2)[1])
    assert loaded.has_unsaved_changes()
    loaded.dialogs.answer = "Cancel"
    widget.save_requested.emit()
    assert loaded.dialogs.asked[-1][0] == "Store changed"
    assert len(loaded.state.pairs) == 2 and loaded.state.revision == 1
    assert loaded.has_unsaved_changes()
    assert host.statuses[-1] == "Not saved: the store changed on disk."
    assert loaded._saving is False

    loaded.dialogs.answer = "Reload"
    widget.save_requested.emit()
    assert loaded.state.revision == 2
    assert [p.id for p in loaded.state.pairs] == [1, 2]
    assert loaded.state.pairs[0].reference_xy == (700.0, 700.0)
    assert loaded.has_unsaved_changes() is False


def test_store_busy(loaded, viewer, widget, monkeypatch):
    click_pair(loaded, viewer, *grid_pairs(1)[0])

    def busy(*args, **kwargs):
        raise StoreBusyError("busy")

    monkeypatch.setattr(controller_module, "save_labelling", busy)
    widget.save_requested.emit()
    assert loaded.dialogs.warned and loaded.dialogs.warned[-1][0] == "Store busy"
    assert loaded._saving is False
    assert loaded.has_unsaved_changes()


def test_save_error_other(loaded, viewer, widget, monkeypatch):
    click_pair(loaded, viewer, *grid_pairs(1)[0])

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(controller_module, "save_labelling", broken)
    widget.save_requested.emit()
    assert loaded.dialogs.warned[-1] == ("Could not save landmarks", "disk full")
    assert loaded._saving is False


def test_local_draft_restore_and_discard(loaded, viewer, widget, host, prepared_store, tmp_path):
    click_pair(loaded, viewer, *grid_pairs(1)[0])
    loaded.flush_pending_updates()
    draft = read_local_draft(prepared_store, host.app_data_dir())
    assert draft is not None and draft["payload"]["revision"] == 1  # placeholder

    v2, h2, _w2, restored = make_env(host.app_data_dir(), answer="Restore")
    assert restored.load_store(prepared_store)
    assert restored.dialogs.asked[0][0] == "Restore local draft?"
    assert [p.id for p in restored.state.pairs] == [1]
    assert restored.has_unsaved_changes()
    assert restored.state.revision is None and restored.state.status == "new"

    v3, h3, _w3, discarded = make_env(host.app_data_dir(), answer="Discard")
    assert discarded.load_store(prepared_store)
    assert discarded.state.pairs == [] and not discarded.has_unsaved_changes()
    assert read_local_draft(prepared_store, host.app_data_dir()) is None


def test_local_draft_with_bad_revision_is_discarded(prepared_store, tmp_path):
    app_dir = tmp_path / "appdata"
    payload = stores.labelling_payload(
        pairs=[{"id": 1, "reference_xy_px": [200.0, 300.0], "moving_xy_px": [210.0, 310.0]}],
        revision=0,
        updated_at="2099-01-01T00:00:00Z",
    )
    write_local_draft(prepared_store, app_dir, payload)
    _v, host, _w, ctrl = make_env(app_dir, answer="Restore")
    assert ctrl.load_store(prepared_store)
    assert any(s.startswith("The local draft could not be read") for s in host.statuses)
    assert ctrl.state.pairs == []
    assert read_local_draft(prepared_store, app_dir) is None


def test_malformed_stored_labelling_never_costs_the_draft(prepared_store, tmp_path):
    app_dir = tmp_path / "appdata"
    pairs = [
        {"id": i + 1, "reference_xy_px": list(ref), "moving_xy_px": list(mov)}
        for i, (ref, mov) in enumerate(grid_pairs(5))
    ]
    write_local_draft(
        prepared_store,
        app_dir,
        stores.labelling_payload(pairs=pairs, revision=1, updated_at="2099-01-01T00:00:00Z"),
    )
    stored = stores.labelling_payload(pairs=pairs[:1], revision=2)
    del stored["coarse"]["offset_um"]  # malformed STORED labelling
    stores.write_root_attr(prepared_store, LABELLING_ATTR, stored)

    # "Restore": the draft comes back
    _v, host, _w, ctrl = make_env(app_dir)

    def ask(title, text, buttons, *, default):
        ctrl.dialogs.asked.append((title, text, tuple(buttons)))
        return "Start fresh" if "Start fresh" in buttons else "Restore"

    ctrl._ask = ask
    assert ctrl.load_store(prepared_store)
    assert [t for t, *_ in ctrl.dialogs.asked] == [
        "Landmarks do not match these images",
        "Restore local draft?",
    ]
    assert [p.id for p in ctrl.state.pairs] == [1, 2, 3, 4, 5]
    assert ctrl.state.revision == 2 and ctrl.has_unsaved_changes()
    assert not any("could not be read" in s for s in host.statuses)

    # "Discard" is the user's explicit choice; only then is the draft deleted
    _v2, _h2, _w2, ctrl2 = make_env(app_dir)

    def ask2(title, text, buttons, *, default):
        return "Start fresh" if "Start fresh" in buttons else "Discard"

    ctrl2._ask = ask2
    assert ctrl2.load_store(prepared_store)
    assert ctrl2.state.pairs == []
    assert read_local_draft(prepared_store, app_dir) is None


def test_stale_local_draft_is_cleared_silently(loaded, viewer, widget, host, prepared_store):
    click_pair(loaded, viewer, *grid_pairs(1)[0])
    widget.save_requested.emit()
    stored = on_disk_labelling(prepared_store)
    write_local_draft(prepared_store, host.app_data_dir(), dict(stored))  # same content
    _v, _h, _w, other = make_env(host.app_data_dir(), answer="Restore")
    assert other.load_store(prepared_store)
    assert other.dialogs.asked == []
    assert read_local_draft(prepared_store, host.app_data_dir()) is None


def test_labelling_shape_mismatch(prepared_store, tmp_path):
    payload = stores.labelling_payload(
        pairs=[{"id": 1, "reference_xy_px": [200.0, 300.0], "moving_xy_px": [210.0, 310.0]}],
        revision=3,
        reference_shape_yx=(1101, 1000),
    )
    stores.write_root_attr(prepared_store, LABELLING_ATTR, payload)
    v, host, widget, ctrl = make_env(tmp_path / "a1", answer="Start fresh")
    assert ctrl.load_store(prepared_store)
    assert ctrl.dialogs.asked[0][0] == "Landmarks do not match these images"
    assert ctrl.state.pairs == [] and ctrl.state.revision == 3
    click_pair(ctrl, v, *grid_pairs(1)[0])
    widget.save_requested.emit()
    assert on_disk_labelling(prepared_store)["revision"] == 4

    stores.write_root_attr(prepared_store, LABELLING_ATTR, payload)
    _v, host2, _w, cancelled = make_env(tmp_path / "a2", answer="Cancel")
    assert cancelled.load_store(prepared_store)
    assert host2.entered == [] and cancelled.is_active() is False
    assert host2.statuses[-1] == "Loading cancelled."


def test_unreadable_stored_labelling_offers_start_fresh(prepared_store, tmp_path):
    payload = stores.labelling_payload(
        pairs=[{"id": 1, "reference_xy_px": [200.0, 300.0], "moving_xy_px": [210.0, 310.0]}],
        revision=2,
    )
    payload["landmarks"]["pairs"][0]["enabled"] = "yes"  # malformed
    stores.write_root_attr(prepared_store, LABELLING_ATTR, payload)
    _v, host, _w, ctrl = make_env(tmp_path / "a", answer="Start fresh")
    assert ctrl.load_store(prepared_store)
    assert ctrl.dialogs.asked[0][0] == "Landmarks do not match these images"
    assert "could not be read" in ctrl.dialogs.asked[0][1]
    assert ctrl.state.revision == 2 and ctrl.state.pairs == []


def test_grid_digest_change_adds_note(prepared_store, tmp_path):
    payload = stores.labelling_payload(
        pairs=[{"id": 1, "reference_xy_px": [200.0, 300.0], "moving_xy_px": [210.0, 310.0]}],
    )
    payload["store_grid_digest"] = "sha256:" + "9" * 64
    stores.write_root_attr(prepared_store, LABELLING_ATTR, payload)
    _v, host, _w, ctrl = make_env(tmp_path / "a")
    assert ctrl.load_store(prepared_store)
    assert "re-prepared" in host.statuses[-1]
    assert [p.id for p in ctrl.state.pairs] == [1]


# ---------------------------------------------------------------------------------------------
# 28 Result mode
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def result_loaded(result_store, tmp_path):
    viewer, host, widget, controller = make_env(tmp_path / "app_result")
    assert controller.load_store(result_store)
    controller.flush_pending_updates()
    return SimpleNamespace(viewer=viewer, host=host, widget=widget, ctrl=controller)


def test_result_mode_overlays_with_crop_offset(result_loaded):
    viewer, widget, ctrl = result_loaded.viewer, result_loaded.widget, result_loaded.ctrl
    assert [layer.name for layer in viewer.layers] == [
        "Reference | DAPI",
        "Reference | AF568",
        "Registered moving | DAPI",
        "Registered moving | AF568",
        "Registered moving | AF647",
        "Landmark-fit positions",
        "Registered moving landmarks",
        "Reference landmarks",
    ]
    assert layer_affine(viewer.layers["Reference | DAPI"]) == pytest.approx(
        np.array([[0.5, 0, 64], [0, 0.5, 32], [0, 0, 1]])
    )
    assert layer_affine(viewer.layers["Registered moving | DAPI"]) == pytest.approx(
        layer_affine(viewer.layers["Reference | DAPI"])
    )
    for layer in viewer.layers:
        assert LAYER_METADATA_KEY in layer.metadata
    ref = viewer.layers["Reference landmarks"]
    assert np.asarray(ref.data_to_world(ref.data[0])) == pytest.approx([150.0, 250.0])
    assert np.asarray(
        viewer.layers["Reference | DAPI"].world_to_data((150.0, 250.0))
    ) == pytest.approx([172.0, 436.0])
    reg = viewer.layers["Registered moving landmarks"]
    assert np.asarray(reg.data_to_world(reg.data[0])) == pytest.approx([149.5, 250.75])
    assert viewer.layers["Landmark-fit positions"].visible is False
    assert [str(s) for s in ref.symbol[:2]] == ["ring", "cross"]
    assert [str(s) for s in reg.symbol] == ["x", "x", "x"]
    assert "SYN1_serial_section" in widget._result_banner.text()
    assert ctrl.has_unsaved_changes() is False
    assert ctrl.session.kind == "result" and ctrl.state is None
    ev = SimpleNamespace(
        type="mouse_press", button=1, position=(1.0, 1.0), pos=np.array([1.0, 1.0])
    )
    assert ctrl.mouse_gesture(ev) is None
    assert result_loaded.host.ready[-1][1] == "result"
    assert viewer.layers.selection.active is viewer.layers["Reference | DAPI"]


def test_result_mode_keys_and_flick(result_loaded):
    viewer, host, ctrl = result_loaded.viewer, result_loaded.host, result_loaded.ctrl
    press_key(viewer, "Control-Z")
    assert host.statuses[-1] == "Nothing to undo in a result store."
    press_key(viewer, "F")
    assert ctrl.display_mode == "reference"
    press_key(viewer, "F")
    assert ctrl.display_mode == "moving"
    assert viewer.layers["Registered moving | DAPI"].opacity == 1
    assert viewer.layers["Reference | DAPI"].opacity == 0
    result_loaded.widget.pair_selected.emit(2)
    assert tuple(viewer.camera.center)[-2:] == pytest.approx((400.25, 350.125), abs=1e-9)


def test_result_overlay_toggles(result_loaded):
    viewer, widget = result_loaded.viewer, result_loaded.widget
    widget.overlay_changed.emit("registered_landmarks", False)
    assert viewer.layers["Registered moving landmarks"].visible is False
    widget.overlay_changed.emit("landmark_fit_positions", True)
    assert viewer.layers["Landmark-fit positions"].visible is True
    widget.overlay_changed.emit("no_such_overlay", True)
    assert widget._overlay_checks["landmark_fit_positions"].isChecked()
    assert widget._overlay_checks["registered_landmarks"].isChecked() is False
    viewer.layers["Reference landmarks"].visible = False
    assert widget._overlay_checks["reference_landmarks"].isChecked() is False
    viewer.layers["Reference landmarks"].visible = True
    assert widget._overlay_checks["reference_landmarks"].isChecked() is True


def test_result_without_result_attr_is_rejected(result_store, host, ctrl):
    import zarr

    group = zarr.open_group(str(result_store), mode="a")
    del group.attrs["merxen_manual_registration_result"]
    assert ctrl.load_store(result_store) is False
    assert host.statuses[-1].startswith("Could not open")


# ---------------------------------------------------------------------------------------------
# 29-33 Lifecycle
# ---------------------------------------------------------------------------------------------


def test_deactivate_cleans_up(loaded, viewer, host, widget):
    widget.landmark_mode_changed.emit(True)
    names = lambda: [f.__name__ for f in viewer.mouse_double_click_callbacks]  # noqa: E731
    assert "double_click_to_zoom" not in names()
    loaded.deactivate()
    assert not [layer for layer in viewer.layers if LAYER_METADATA_KEY in layer.metadata]
    for key in ("F", "Shift-F", "Control-Z", "Escape", "Control-Alt-T", "Control-Delete"):
        assert KeyBinding.from_str(key) not in viewer.keymap
    assert "double_click_to_zoom" in names()
    viewer.camera.zoom = 3.0  # raises nothing
    assert host.left == 1 and loaded.is_active() is False
    loaded.deactivate()
    assert host.left == 1


def test_deactivate_writes_draft_and_restores_keys(loaded, viewer, host, prepared_store):

    def previous(v):
        return None

    v2, _h2, _w2, c2 = make_env(host.app_data_dir())
    v2.bind_key("F", previous, overwrite=True)
    assert c2.load_store(prepared_store)
    assert v2.keymap[KeyBinding.from_str("F")] is not previous
    click_pair(c2, v2, *grid_pairs(1)[0])
    c2.deactivate()
    assert v2.keymap[KeyBinding.from_str("F")] is previous
    assert read_local_draft(prepared_store, host.app_data_dir()) is not None


def test_confirm_discard_unsaved(loaded, viewer, host, prepared_store):
    assert loaded.confirm_discard_unsaved() is True and loaded.dialogs.asked == []
    click_pair(loaded, viewer, *grid_pairs(1)[0])
    loaded.dialogs.answer = "Cancel"
    assert loaded.confirm_discard_unsaved() is False
    assert loaded.dialogs.asked[-1][0] == "Unsaved landmarks"
    loaded.dialogs.answer = "Discard"
    assert loaded.confirm_discard_unsaved() is True
    assert read_local_draft(prepared_store, host.app_data_dir()) is not None
    loaded.dialogs.answer = "Save draft"
    assert loaded.confirm_discard_unsaved() is True
    assert on_disk_labelling(prepared_store)["revision"] == 1
    assert loaded.has_unsaved_changes() is False


def test_confirm_discard_save_error_returns_false(loaded, viewer, monkeypatch):
    click_pair(loaded, viewer, *grid_pairs(1)[0])
    monkeypatch.setattr(
        controller_module,
        "save_labelling",
        lambda *a, **k: (_ for _ in ()).throw(StoreBusyError("x")),
    )
    loaded.dialogs.answer = "Save draft"
    assert loaded.confirm_discard_unsaved() is False
    assert loaded.dialogs.warned[-1][0] == "Store busy"


def test_load_rejects_non_registration_store(ctrl, host, tmp_path):
    import spatialdata as sd
    from spatialdata.models import Image2DModel

    plain = tmp_path / "plain.zarr"
    image = Image2DModel.parse(np.zeros((1, 32, 32), dtype=np.uint16), dims=("c", "y", "x"))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sd.SpatialData(images={"img": image}).write(str(plain))
    assert ctrl.load_store(plain) is False
    assert host.worker_keys == []
    assert host.statuses[-1].startswith("Could not open")
    assert ctrl.load_store(tmp_path / "missing.zarr") is False
    assert host.statuses[-1].startswith("Registration store not found")


def test_cancel_background_drops_stale_load(viewer, tmp_path, prepared_store):
    _v, host, _w, ctrl = make_env(tmp_path / "app", deferred=True)
    assert ctrl.load_store(prepared_store)
    ctrl.cancel_background()
    host.run_pending()
    assert ctrl.is_active() is False
    assert host.entered == []
    assert [layer for layer in _v.layers] == []
    assert not host.statuses or host.statuses[-1] == "Loading cancelled."


def test_prepare_session_cancel_raises(host, ctrl, prepared_store):
    with pytest.raises(RegistrationLoadCancelled):
        ctrl.prepare_session(prepared_store, lambda: True)


def test_prepare_session_forwards_cancel_check_to_the_contrast_read(
    host, ctrl, prepared_store, monkeypatch
):
    received = []

    def spy(level0, coarse, index, **kwargs):
        received.append(kwargs.get("cancel_check"))
        return (0.0, 1.0)

    monkeypatch.setattr(controller_module, "contrast_limits_from_tissue_tiles", spy)
    marker = lambda: False  # noqa: E731
    ctrl.prepare_session(prepared_store, marker)
    assert received and all(check is marker for check in received)


def test_newer_load_supersedes_older(prepared_store, tmp_path, prepared_template):
    second = tmp_path / "second_prepared.zarr"
    shutil.copytree(prepared_template, second)
    _v, host, _w, ctrl = make_env(tmp_path / "app", deferred=True)
    assert ctrl.load_store(prepared_store)
    assert ctrl.load_store(second)
    host.run_pending()
    assert host.entered == [second]


def test_save_flag_survives_store_switch(prepared_store, tmp_path, prepared_template):
    store_b = tmp_path / "B_prepared.zarr"
    shutil.copytree(prepared_template, store_b)

    # (a) the host cancels the queued save of A, then B loads
    v, host, widget, ctrl = make_env(tmp_path / "app_a", deferred=True)
    assert ctrl.load_store(prepared_store)
    host.run_pending()
    click_pair(ctrl, v, *grid_pairs(1)[0])
    widget.save_requested.emit()
    assert ctrl._saving is True and host.worker_keys[-1] == "registration_save"
    assert ctrl.load_store(store_b)
    assert host.drop_pending(keys=["registration_save"]) == 1
    host.run_pending()
    assert ctrl.session.path == store_b
    assert ctrl._saving is False and widget._busy is False
    assert widget._finish_reason != "Saving…"
    click_pair(ctrl, v, *grid_pairs(1)[0])
    widget.save_requested.emit()
    assert host.worker_keys[-1] == "registration_save"
    assert host.statuses[-1] != "A save is already running."
    assert len(host.pending) == 1
    host.run_pending()
    assert on_disk_labelling(store_b)["revision"] == 1
    assert on_disk_labelling(prepared_store) is None

    # (b) B loads first, then the stale save of A runs
    store_c = tmp_path / "C_prepared.zarr"
    shutil.copytree(prepared_template, store_c)
    v, host, widget, ctrl = make_env(tmp_path / "app_b", deferred=True)
    assert ctrl.load_store(store_c)
    host.run_pending()
    click_pair(ctrl, v, *grid_pairs(1)[0])
    widget.save_requested.emit()
    store_d = tmp_path / "D_prepared.zarr"
    shutil.copytree(prepared_template, store_d)
    assert ctrl.load_store(store_d)
    host.run_pending(keys=["registration_load"])
    assert ctrl.session.path == store_d
    host.run_pending(keys=["registration_save"])
    assert on_disk_labelling(store_c)["revision"] == 1
    assert ctrl._saving is False and widget._busy is False
    assert ctrl.dialogs.asked == [] and ctrl.dialogs.warned == []
    assert ctrl.state.revision is None


def test_second_save_while_running_is_refused(prepared_store, tmp_path):
    v, host, widget, ctrl = make_env(tmp_path / "app", deferred=True)
    assert ctrl.load_store(prepared_store)
    host.run_pending()
    click_pair(ctrl, v, *grid_pairs(1)[0])
    widget.save_requested.emit()
    widget.save_requested.emit()
    assert host.statuses[-1] == "A save is already running."
    assert len(host.pending) == 1
    host.run_pending()
    assert ctrl._saving is False and ctrl.state.revision == 1


def test_save_draft_choice_refused_while_an_async_save_runs(prepared_store, tmp_path):
    v, host, widget, ctrl = make_env(tmp_path / "app", deferred=True, answer="Save draft")
    assert ctrl.load_store(prepared_store)
    host.run_pending()
    click_pair(ctrl, v, *grid_pairs(1)[0])
    widget.save_requested.emit()
    assert ctrl._saving is True
    assert ctrl.confirm_discard_unsaved() is False  # the "Save draft" choice
    assert host.statuses[-1] == "A save is already running."
    assert ctrl.dialogs.warned == []
    assert [t for t, *_ in ctrl.dialogs.asked] == ["Unsaved landmarks"]  # no "Store changed"
    host.run_pending()
    assert ctrl._saving is False and ctrl.state.revision == 1
    assert ctrl.dialogs.warned == []


def test_adopt_fit_sets_coarse_from_robust_fit(loaded, viewer, widget, host):
    widget.adopt_fit.emit()
    assert host.statuses[-1] == "Add at least 3 enabled pairs first."
    place_truth_pairs(loaded, viewer)
    widget.adopt_fit.emit()
    loaded.flush_pending_updates()
    coarse = loaded.state.coarse
    assert coarse.rotation_deg == pytest.approx(12.0, abs=0.05)
    assert coarse.offset_um == pytest.approx((5.0, -2.5), abs=0.05)
    assert host.statuses[-1].startswith("Coarse alignment set from the fit")
    assert layer_affine(viewer.layers["Moving | DAPI"]) == pytest.approx(
        coarse_affine(coarse), abs=1e-9
    )


def test_preview_needs_three_pairs(loaded, viewer, widget, host):
    widget.preview_changed.emit(True)
    assert host.statuses[-1] == "Preview needs at least 3 enabled pairs."
    place_truth_pairs(loaded, viewer, MOVING_POINTS[:3])
    widget.preview_changed.emit(True)
    assert loaded._preview is True
    widget.delete_pairs.emit([1])
    assert loaded._preview is False  # fewer than 3 enabled pairs turn the preview off


def test_pair_selected_centres_camera(loaded, viewer, widget):
    click_pair(loaded, viewer, (400.0, 500.0), (410.0, 510.0))
    widget.display_mode_changed.emit("reference")
    widget.pair_selected.emit(1)
    centre = tuple(viewer.camera.center)[-2:]
    assert centre == pytest.approx((250.0, 200.0))
    widget.display_mode_changed.emit("moving")
    widget.pair_selected.emit(1)
    world = viewer.layers["Moving landmarks"].data_to_world((510.0, 410.0))
    assert tuple(viewer.camera.center)[-2:] == pytest.approx(tuple(world))


def test_slots_without_session_report(ctrl, host, widget):
    widget.rotate_by.emit(5.0)
    assert host.statuses[-1] == "Load a registration store first."
    widget.landmark_mode_changed.emit(True)
    assert host.statuses[-1] == "Load a registration store first."
    assert ctrl.mouse_gesture(SimpleNamespace(type="mouse_press", button=1)) is None
    assert ctrl.has_unsaved_changes() is False and ctrl.confirm_discard_unsaved() is True


def test_set_widget_twice_disconnects_old(loaded, host):
    old = loaded._widget
    new = ManualRegistrationWidget()
    loaded.set_widget(new)
    old.rotate_by.emit(7.0)
    assert loaded.state.coarse.rotation_deg == 0.0
    new.rotate_by.emit(7.0)
    assert loaded.state.coarse.rotation_deg == 7.0
    assert new.display_mode() == "both"


def test_replacing_a_session_stops_the_old_timers(
    loaded, viewer, widget, host, prepared_template, tmp_path
):
    store_b = tmp_path / "B_prepared.zarr"
    shutil.copytree(prepared_template, store_b)
    click_pair(loaded, viewer, *grid_pairs(1)[0])
    first = loaded.session.path
    assert loaded._draft_timer is not None and loaded._draft_timer.isActive()
    assert loaded.load_store(store_b)  # no prompt here; VW3 asks first
    assert loaded.session.path == store_b
    for timer in (loaded._draft_timer, loaded._affine_timer):
        assert timer is None or not timer.isActive()
    assert read_local_draft(first, host.app_data_dir()) is not None  # A's draft was written
    assert read_local_draft(store_b, host.app_data_dir()) is None  # ... and B has none


def test_failed_apply_before_teardown_keeps_the_old_session(
    loaded, viewer, host, monkeypatch, prepared_template, tmp_path
):
    store_b = tmp_path / "B_prepared.zarr"
    shutil.copytree(prepared_template, store_b)
    click_pair(loaded, viewer, *grid_pairs(1)[0])
    old = loaded.session
    left = host.left

    def boom(*args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(loaded, "_decide_draft", boom)
    write_local_draft(
        store_b,
        host.app_data_dir(),
        stores.labelling_payload(
            pairs=[{"id": 1, "reference_xy_px": [200.0, 300.0], "moving_xy_px": [210.0, 310.0]}],
            updated_at="2099-01-01T00:00:00Z",
        ),
    )
    assert loaded.load_store(store_b)
    assert loaded.session is old and host.left == left  # nothing was torn down
    assert "Could not load registration store" in host.statuses[-1]
    assert [layer.name for layer in viewer.layers].count("Reference | DAPI") == 1
    assert ("end", "registration", None) in host.progress


def test_set_widget_reads_the_auto_switch_checkbox(host):
    widget = ManualRegistrationWidget()
    widget._auto_switch_check.setChecked(False)
    assert widget.auto_switch() is False
    controller = ManualRegistrationController(host)
    controller.set_widget(widget)
    assert controller._auto_switch is False
    widget._auto_switch_check.setChecked(True)
    assert controller._auto_switch is True


def test_reload_over_active_session_keeps_clean_state(loaded, viewer, widget, host, prepared_store):
    click_pair(loaded, viewer, *grid_pairs(1)[0])
    assert loaded.load_store(prepared_store)  # no prompt here; VW3 asks first
    assert len(loaded.state.pairs) == 0
    assert loaded.dialogs.asked == []
    assert host.entered == [prepared_store, prepared_store]
    assert [layer.name for layer in viewer.layers].count("Reference | DAPI") == 1


# ---------------------------------------------------------------------------------------------
# 33 Builders against the golden examples
# ---------------------------------------------------------------------------------------------


def _find_schema(node, schema, kind=None):
    if isinstance(node, dict):
        if node.get("schema") == schema and (kind is None or node.get("kind") == kind):
            return node
        for value in node.values():
            found = _find_schema(value, schema, kind)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_schema(value, schema, kind)
            if found is not None:
                return found
    return None


def _same_keys(ours, golden, path="$"):
    if isinstance(ours, dict) and isinstance(golden, dict):
        assert set(ours) == set(golden), f"{path}: {set(ours) ^ set(golden)}"
        for key in ours:
            _same_keys(ours[key], golden[key], f"{path}.{key}")


def test_store_builders_match_golden_examples(prepared_store):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    prepared = _find_schema(golden, "merxen.manual_registration.store", "prepared")
    labelling = _find_schema(golden, "merxen.manual_registration.labelling")
    result = _find_schema(golden, "merxen.manual_registration.store", "result")
    assert prepared is not None and labelling is not None and result is not None
    _same_keys(stores.prepared_store_attr(), prepared)
    _same_keys(stores.result_store_attr(), result)
    pair = {"id": 1, "reference_xy_px": [200.0, 300.0], "moving_xy_px": [210.0, 310.0]}
    payload = stores.labelling_payload(pairs=[pair, {**pair, "id": 2, "enabled": False}])
    _same_keys(payload, labelling)
    descriptor = parse_store_descriptor(prepared_store, read_root_attrs(prepared_store))
    state = labelling_from_attrs(payload, descriptor)
    assert [p.id for p in state.pairs] == [1, 2] and state.pairs[1].enabled is False
    assert state.load_warnings == ()


# ---------------------------------------------------------------------------------------------
# MerXen landmark check (C3), preview result stores (C4), field banner (C5)
# ---------------------------------------------------------------------------------------------

CHECK_HEADERS = ["#", "", "On", "LOO µm", "p", "Weight", "Image", "Flags"]
RIGID_HEADERS = ["#", "", "On", "Rigid fit µm", "Rigid LOO µm", "Flags"]


def labelled_store(path, *, n_pairs=10, revision=2):
    """Write a labelling of ``n_pairs`` pairs into a prepared store; return its payload."""
    pairs = [
        {"id": i + 1, "reference_xy_px": ref, "moving_xy_px": mov}
        for i, (ref, mov) in enumerate(grid_pairs(n_pairs))
    ]
    payload = stores.labelling_payload(pairs=pairs, revision=revision, status="finished")
    stores.write_root_attr(path, LABELLING_ATTR, payload)
    return payload


def matching_check(payload, **kwargs):
    return stores.check_payload(
        labelling_content_sha256=payload["content_sha256"],
        pair_ids=[p["id"] for p in payload["landmarks"]["pairs"]],
        **kwargs,
    )


def table_headers(widget):
    table = widget._table
    return [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]


@pytest.fixture
def labelled(prepared_store):
    payload = labelled_store(prepared_store)
    return SimpleNamespace(path=prepared_store, payload=payload)


def load(ctrl, path):
    assert ctrl.load_store(path)
    ctrl.flush_pending_updates()
    return ctrl


def test_check_attr_with_matching_digests_shows_the_check(ctrl, widget, labelled):
    stores.write_check_attr(
        labelled.path,
        matching_check(labelled.payload, classes={2: "likely_misclick", 3: "check"}),
    )
    load(ctrl, labelled.path)
    assert ctrl.state.loaded_content_sha == labelled.payload["content_sha256"]
    assert table_headers(widget) == CHECK_HEADERS
    table = widget._table
    # pair 1: p and weight are the lower of the final and landmarks-only fields
    assert [table.item(0, c).text() for c in (0, 3, 4, 5, 6, 7)] == [
        "1",
        "23.50",
        "0.41",
        "1.12",
        "confirmed",
        "",
    ]
    assert table.item(1, 7).text() == "⚠ likely misclick"
    assert table.item(2, 7).text() == "? check"
    status = widget._check_status_label.text()
    assert status.startswith("Robust check (store, 2026-11-18T09:00:00Z, cortex_serial):")
    assert (
        "LOO median 34 µm (clicks only 42 µm), p90 97 µm; 1 to check, 1 likely misclicks" in status
    )
    assert "12 tissue windows." in status
    assert ctrl.has_unsaved_changes() is False


def test_without_a_check_the_table_is_the_rigid_fit(ctrl, widget, labelled):
    load(ctrl, labelled.path)
    assert table_headers(widget) == RIGID_HEADERS
    text = widget._check_status_label.text()
    assert text.startswith("No robust check for these landmarks.")
    assert "rigid (similarity) fit, which cannot bend" in text
    assert "merxen manual-registration-check" in text
    assert widget._summary_label.text().startswith("Rigid (similarity) fit on")


def test_load_check_file_shows_the_check_with_the_file_name(ctrl, widget, labelled, tmp_path):
    load(ctrl, labelled.path)
    check_file = stores.write_check_file(
        tmp_path / "SYN1_cortex_serial_check.json",
        matching_check(labelled.payload, with_details=True),
    )
    widget.check_file_chosen.emit(str(check_file))
    assert table_headers(widget) == CHECK_HEADERS
    assert "Robust check (SYN1_cortex_serial_check.json," in widget._check_status_label.text()
    assert widget._check_dialog_dir == str(tmp_path)
    assert ctrl.has_unsaved_changes() is False


def test_stale_check_file_keeps_rigid_mode_until_refreshed(ctrl, widget, labelled, tmp_path):
    load(ctrl, labelled.path)
    stale = matching_check(labelled.payload)
    stale["labelling_content_sha256"] = "sha256:" + "9" * 64
    check_file = stores.write_check_file(tmp_path / "SYN1_cortex_serial_check.json", stale)
    widget.check_file_chosen.emit(str(check_file))
    assert table_headers(widget) == RIGID_HEADERS
    assert widget._check_status_label.text().startswith(
        "The landmarks, crops or coarse alignment changed since the check was run (revision 3)."
    )
    # a newer file of the same name; Refresh reads it again
    stores.write_check_file(check_file, matching_check(labelled.payload))
    widget.refresh_check_requested.emit()
    assert table_headers(widget) == CHECK_HEADERS
    assert "SYN1_cortex_serial_check.json" in widget._check_status_label.text()


def test_check_for_another_preparation_is_not_applied(ctrl, widget, labelled):
    stores.write_check_attr(
        labelled.path, matching_check(labelled.payload, store_content_digest="sha256:" + "7" * 64)
    )
    load(ctrl, labelled.path)
    assert table_headers(widget) == RIGID_HEADERS
    assert "another preparation of the images" in widget._check_status_label.text()


def test_editing_a_pair_makes_the_check_stale(ctrl, widget, labelled):
    stores.write_check_attr(labelled.path, matching_check(labelled.payload))
    load(ctrl, labelled.path)
    assert table_headers(widget) == CHECK_HEADERS
    widget.pair_enabled_changed.emit(1, False)
    assert table_headers(widget) == RIGID_HEADERS
    assert (
        "The landmarks, crops or coarse alignment changed since the check was run"
        in widget._check_status_label.text()
    )
    widget.pair_enabled_changed.emit(1, True)  # undone by hand: the digests match again
    assert table_headers(widget) == CHECK_HEADERS
    # adding a pair changes the labelling too
    ctrl._widget.landmark_mode_changed.emit(True)
    ctrl.state.add_pair((10.0, 20.0), (30.0, 40.0))
    ctrl._content_changed()
    assert table_headers(widget) == RIGID_HEADERS


def test_refresh_reads_the_store_attribute_without_touching_the_labelling(
    ctrl, widget, labelled, host
):
    load(ctrl, labelled.path)
    assert table_headers(widget) == RIGID_HEADERS
    state_before = ctrl.state
    revision_before = state_before.revision
    sha_before = state_before.content_sha256()
    stores.write_check_attr(labelled.path, matching_check(labelled.payload))
    widget.refresh_check_requested.emit()
    assert host.worker_keys[-1] == "registration_check"
    assert table_headers(widget) == CHECK_HEADERS
    assert "Robust check (store," in widget._check_status_label.text()
    assert ctrl.state is state_before
    assert (state_before.revision, state_before.content_sha256()) == (revision_before, sha_before)
    assert ctrl.has_unsaved_changes() is False
    assert ctrl.session.descriptor.check is not None
    # the labelling on disk is exactly what it was
    assert on_disk_labelling(labelled.path) == labelled.payload
    # removing the attribute and refreshing returns to the rigid fit
    import zarr

    del zarr.open_group(str(labelled.path), mode="a").attrs["merxen_manual_registration_check"]
    widget.refresh_check_requested.emit()
    assert table_headers(widget) == RIGID_HEADERS


def test_refresh_picks_up_a_new_store_content_digest(ctrl, widget, labelled):
    stores.write_check_attr(
        labelled.path, matching_check(labelled.payload, store_content_digest="sha256:" + "7" * 64)
    )
    load(ctrl, labelled.path)
    assert table_headers(widget) == RIGID_HEADERS
    attr = read_root_attrs(labelled.path)[STORE_ATTR]
    stores.write_root_attr(
        labelled.path, STORE_ATTR, {**attr, "content_digest": "sha256:" + "7" * 64}
    )
    widget.refresh_check_requested.emit()
    assert table_headers(widget) == CHECK_HEADERS


def test_malformed_store_attribute_gives_a_note_and_the_rigid_fit(ctrl, widget, labelled, host):
    stores.write_check_attr(labelled.path, {"schema": "merxen.manual_registration.check"})
    load(ctrl, labelled.path)
    assert table_headers(widget) == RIGID_HEADERS
    assert "the landmark check could not be read:" in widget._check_status_label.text()
    assert "the landmark check could not be read:" in host.statuses[-1]
    assert widget._table.rowCount() == 10


def test_newer_check_schema_gives_an_update_note(ctrl, widget, labelled, host):
    stores.write_check_attr(
        labelled.path, {**matching_check(labelled.payload), "schema_version": 2}
    )
    load(ctrl, labelled.path)
    assert table_headers(widget) == RIGID_HEADERS
    assert "written by a newer MerXen" in widget._check_status_label.text()
    assert "update this viewer" in host.statuses[-1]


def test_malformed_or_missing_check_file_keeps_the_previous_check(ctrl, widget, labelled, tmp_path):
    stores.write_check_attr(labelled.path, matching_check(labelled.payload))
    load(ctrl, labelled.path)
    assert table_headers(widget) == CHECK_HEADERS
    for name, content in (("bad.json", "{not json"), ("list.json", "[]"), ("empty.json", "{}")):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        widget.check_file_chosen.emit(str(path))
        assert table_headers(widget) == CHECK_HEADERS  # the previous check stays
        assert "the landmark check could not be read:" in widget._check_status_label.text()
    widget.check_file_chosen.emit(str(tmp_path / "missing_check.json"))
    assert table_headers(widget) == CHECK_HEADERS
    assert "could not be read" in widget._check_status_label.text()
    # a good file clears the note
    good = stores.write_check_file(tmp_path / "good_check.json", matching_check(labelled.payload))
    widget.check_file_chosen.emit(str(good))
    assert "could not be read" not in widget._check_status_label.text()
    newer = tmp_path / "newer_check.json"
    newer.write_text(
        json.dumps({**matching_check(labelled.payload), "schema_version": 2}), encoding="utf-8"
    )
    widget.check_file_chosen.emit(str(newer))
    assert "written by a newer MerXen" in widget._check_status_label.text()
    assert table_headers(widget) == CHECK_HEADERS


def test_check_workers_are_disabled_while_running_and_stale_results_are_ignored(
    prepared_store, tmp_path
):
    payload = labelled_store(prepared_store)
    viewer, host, widget, controller = make_env(tmp_path / "app_deferred", deferred=True)
    assert controller.load_store(prepared_store)
    host.run_pending()
    controller.flush_pending_updates()
    check_file = stores.write_check_file(
        tmp_path / "SYN1_cortex_serial_check.json", matching_check(payload)
    )
    widget.check_file_chosen.emit(str(check_file))
    assert host.worker_keys[-1] == "registration_check" and len(host.pending) == 1
    assert not widget._load_check_button.isEnabled()
    assert not widget._refresh_check_button.isEnabled()
    widget.refresh_check_requested.emit()  # ignored while a worker runs
    assert len(host.pending) == 1
    assert table_headers(widget) == RIGID_HEADERS
    host.run_pending()
    assert table_headers(widget) == CHECK_HEADERS
    assert widget._load_check_button.isEnabled() and widget._refresh_check_button.isEnabled()

    # a read that finishes after the session was replaced is dropped
    widget.refresh_check_requested.emit()
    assert len(host.pending) == 1
    assert controller.load_store(prepared_store)
    host.drop_pending(keys=["registration_check"])
    host.run_pending()
    controller.flush_pending_updates()
    assert table_headers(widget) == RIGID_HEADERS  # the new session starts without a file check
    assert widget._load_check_button.isEnabled()


def test_a_late_check_result_of_an_earlier_session_is_ignored(prepared_store, tmp_path):
    payload = labelled_store(prepared_store)
    viewer, host, widget, controller = make_env(tmp_path / "app_late", deferred=True)
    assert controller.load_store(prepared_store)
    host.run_pending()
    check_file = stores.write_check_file(tmp_path / "x_check.json", matching_check(payload))
    widget.check_file_chosen.emit(str(check_file))
    late = host.pending[-1]
    host.pending.remove(late)
    assert controller.load_store(prepared_store)
    host.run_pending()
    late[1](late[0]())  # the old session's worker calls back
    assert table_headers(widget) == RIGID_HEADERS
    assert controller._check is None


def test_a_late_store_check_refresh_of_an_earlier_session_is_ignored(prepared_store, tmp_path):
    payload = labelled_store(prepared_store)
    viewer, host, widget, controller = make_env(tmp_path / "app_late_store", deferred=True)
    assert controller.load_store(prepared_store)
    host.run_pending()
    stores.write_check_attr(prepared_store, matching_check(payload))
    widget.refresh_check_requested.emit()  # no file loaded: re-reads the store attribute
    late = host.pending[-1]
    assert host.worker_keys[-1] == "registration_check"
    host.pending.remove(late)
    late_attrs = late[0]()  # the old read saw the check
    zarr_json = prepared_store / "zarr.json"  # the next session starts without the check
    document = json.loads(zarr_json.read_text())
    document["attributes"].pop("merxen_manual_registration_check")
    zarr_json.write_text(json.dumps(document))
    assert controller.load_store(prepared_store)
    host.run_pending()
    descriptor_before = controller.session.descriptor
    late[1](late_attrs)
    assert controller.session.descriptor is descriptor_before
    assert controller._check is None
    assert table_headers(widget) == RIGID_HEADERS


def test_cancelling_background_work_frees_the_check_buttons(prepared_store, tmp_path):
    payload = labelled_store(prepared_store)
    viewer, host, widget, controller = make_env(tmp_path / "app_cancel", deferred=True)
    assert controller.load_store(prepared_store)
    host.run_pending()
    check_file = stores.write_check_file(tmp_path / "x_check.json", matching_check(payload))
    widget.check_file_chosen.emit(str(check_file))
    assert not widget._load_check_button.isEnabled()
    controller.cancel_background()
    host.drop_pending(keys=["registration_check"])
    assert widget._load_check_button.isEnabled()
    assert widget._refresh_check_button.isEnabled()
    widget.refresh_check_requested.emit()
    assert host.worker_keys[-1] == "registration_check" and len(host.pending) == 1


def test_the_check_is_forgotten_when_the_session_ends(ctrl, widget, labelled):
    stores.write_check_attr(labelled.path, matching_check(labelled.payload))
    load(ctrl, labelled.path)
    assert ctrl._check is not None
    ctrl.deactivate()
    assert ctrl._check is None and ctrl._check_raw is None


def test_suggestions_layer_is_hidden_and_follows_the_check(ctrl, viewer, widget, labelled):
    stores.write_check_attr(labelled.path, matching_check(labelled.payload))
    load(ctrl, labelled.path)
    layer = viewer.layers["Suggested landmarks"]
    assert layer.visible is False
    assert layer.metadata[LAYER_METADATA_KEY]["role"] == "reference"
    assert layer.data.shape == (2, 2)
    # napari (row, col) = (y, x) of the reference px; ranks are the labels
    assert layer.data.tolist() == [[9000.5, 6000.0], [41000.0, 52000.0]]
    assert list(layer.features["label"]) == ["1", "2"]
    assert [str(s) for s in layer.symbol] == ["diamond", "diamond"]
    assert to_hex(layer.face_color[0]) == "#ffd400"
    assert layer_affine(layer) == pytest.approx(layer_affine(viewer.layers["Reference landmarks"]))
    assert widget._suggestions_check.isEnabled() and not widget._suggestions_check.isChecked()
    widget._suggestions_check.setChecked(True)
    assert layer.visible is True
    layer.visible = False  # the layer's own eye is mirrored by the checkbox
    assert widget._suggestions_check.isChecked() is False
    widget._suggestions_check.setChecked(True)
    # a stale check empties the layer and disables the checkbox
    widget.pair_enabled_changed.emit(1, False)
    assert len(layer.data) == 0
    assert not widget._suggestions_check.isEnabled()
    widget.pair_enabled_changed.emit(1, True)
    assert len(layer.data) == 2


def test_suggestions_layer_is_empty_without_a_check(ctrl, viewer, widget, labelled):
    load(ctrl, labelled.path)
    assert len(viewer.layers["Suggested landmarks"].data) == 0
    assert not widget._suggestions_check.isEnabled()


PREVIEW = {"mode": "preview", "level_factor": 4, "pixel_size_um": 2.0}


def test_preview_result_store_is_placed_by_pixel_centre(result_template, tmp_path):
    path = tmp_path / "SYN1_preview.zarr"
    stores.build_result_store(path, materialization=PREVIEW)
    viewer, host, widget, controller = make_env(tmp_path / "app_preview")
    assert controller.load_store(path)
    controller.flush_pending_updates()
    p, factor = P_R, 4
    crop = stores.RESULT_CROP
    half = (factor - 1) / 2
    expected = np.array(
        [
            [p * factor, 0, p * (crop["y0"] + half)],
            [0, p * factor, p * (crop["x0"] + half)],
            [0, 0, 1],
        ]
    )
    for name in ("Reference | DAPI", "Registered moving | DAPI", "Registered moving | AF647"):
        assert layer_affine(viewer.layers[name]) == pytest.approx(expected)
    # the overlays stay diag(p) on full-frame px
    for name in ("Reference landmarks", "Registered moving landmarks", "Landmark-fit positions"):
        assert layer_affine(viewer.layers[name]) == pytest.approx(np.diag([p, p, 1.0]))
    ref = viewer.layers["Reference landmarks"]
    assert np.asarray(ref.data_to_world(ref.data[0])) == pytest.approx([150.0, 250.0])


def test_result_store_without_materialization_keeps_todays_placement(result_loaded):
    viewer = result_loaded.viewer
    assert layer_affine(viewer.layers["Reference | DAPI"]) == pytest.approx(
        np.array([[0.5, 0, 64], [0, 0.5, 32], [0, 0, 1]])
    )


def test_result_store_with_a_malformed_materialization_is_refused(tmp_path, host, ctrl):
    path = tmp_path / "SYN1_bad.zarr"
    stores.build_result_store(path, materialization={"mode": "preview", "level_factor": 0})
    assert ctrl.load_store(path) is False
    assert "materialization is missing or invalid" in host.statuses[-1]


FIELD = {
    "model": "robust_gp",
    "loo_um": {"median": 34.5, "p90": 97.25, "max": 183.5},
    "class_counts": {"check": 1, "likely_misclick": 2, "ok": 5},
    "materialize": {"mode": "preview", "level_factor": 4, "pixel_size_um": 2.0},
}


def test_result_banner_has_a_field_line_for_field_runs():
    result = stores.result_attr(landmarks=stores.default_result_landmarks())
    assert len(result_banner(result).splitlines()) == 3
    result["field"] = None
    assert len(result_banner(result).splitlines()) == 3
    result["field"] = FIELD
    lines = result_banner(result).splitlines()
    assert len(lines) == 4
    assert lines[3] == (
        "Field: robust GP · LOO median 34.50 µm (p90 97.25) · 1 to check, 2 likely misclicks · "
        "preview 2.00 µm/px"
    )
    result["field"] = {"model": "robust_gp"}
    assert result_banner(result).splitlines()[3] == (
        "Field: robust GP · LOO median n/a µm (p90 n/a) · n/a to check, n/a likely misclicks · "
        "n/a n/a µm/px"
    )


def test_result_store_shows_the_field_line(tmp_path):
    path = tmp_path / "SYN1_field.zarr"
    stores.build_result_store(path, materialization=PREVIEW, field=FIELD)
    viewer, host, widget, controller = make_env(tmp_path / "app_field")
    assert controller.load_store(path)
    assert "Field: robust GP · LOO median 34.50 µm" in widget._result_banner.text()


def _bright_tail_pyramid(seed: int = 3):
    """Return (level0, coarse, true level-0 tissue values): dim tissue, sparse bright objects."""
    import xarray as xr

    rng = np.random.default_rng(seed)
    size, block = 2048, 64
    full = np.zeros((1, size, size), dtype=np.float32)
    full[0, 256:-256, 256:-256] = rng.normal(100, 10, (size - 512, size - 512)).clip(1)
    # Small bright objects (4 x 4 px): about 1.5 % of tissue pixels at 1500, 0.3 % at 5000.
    for value, count in ((1500, 3300), (5000, 650)):
        ys = rng.integers(260, size - 270, count)
        xs = rng.integers(260, size - 270, count)
        for dy in range(4):
            for dx in range(4):
                full[0, ys + dy, xs + dx] = value
    coarse = full.reshape(1, size // block, block, size // block, block).mean(axis=(2, 4))
    level0 = xr.DataArray(full, dims=("c", "y", "x"))
    coarse_array = xr.DataArray(coarse, dims=("c", "y", "x"))
    return level0, coarse_array, full[0][full[0] != 0]


def test_tissue_tile_contrast_recovers_bright_tail_the_coarse_level_loses():
    level0, coarse, tissue = _bright_tail_pyramid()
    true_high = float(np.percentile(tissue, 99.9))
    old_high = contrast_limits_from_level(coarse, 0)[1]
    assert old_high < float(np.percentile(tissue, 99))
    low, high = contrast_limits_from_tissue_tiles(level0, coarse, 0, n_tiles=48)
    assert high == pytest.approx(true_high, rel=0.1)
    assert 0 < low < high


def test_tissue_tile_contrast_is_deterministic_and_seed_tolerant():
    level0, coarse, _ = _bright_tail_pyramid()
    first = contrast_limits_from_tissue_tiles(level0, coarse, 0, n_tiles=48)
    assert contrast_limits_from_tissue_tiles(level0, coarse, 0, n_tiles=48) == first
    other = contrast_limits_from_tissue_tiles(level0, coarse, 0, n_tiles=48, seed=5)
    assert other[1] == pytest.approx(first[1], rel=0.1)
    assert other[0] == pytest.approx(first[0], rel=0.1)


def test_tissue_tile_contrast_ignores_padding_and_empty_channels():
    import xarray as xr

    zeros = xr.DataArray(np.zeros((1, 600, 600), dtype=np.uint16), dims=("c", "y", "x"))
    coarse = xr.DataArray(np.zeros((1, 30, 30), dtype=np.uint16), dims=("c", "y", "x"))
    assert contrast_limits_from_tissue_tiles(zeros, coarse, 0) == (0.0, 1.0)
    # Coarse level claims tissue, but the level-0 tiles are all padding.
    coarse_tissue = xr.DataArray(np.ones((1, 30, 30), dtype=np.uint16), dims=("c", "y", "x"))
    assert contrast_limits_from_tissue_tiles(zeros, coarse_tissue, 0) == (0.0, 1.0)
    constant = xr.DataArray(np.full((1, 600, 600), 9, dtype=np.uint16), dims=("c", "y", "x"))
    assert contrast_limits_from_tissue_tiles(constant, coarse_tissue, 0) == (9.0, 10.0)


def test_tissue_tile_contrast_stops_when_cancelled_between_tiles():
    level0, coarse, _ = _bright_tail_pyramid()
    calls = []

    def cancel_check() -> bool:
        calls.append(1)
        return len(calls) > 2

    with pytest.raises(RegistrationLoadCancelled):
        contrast_limits_from_tissue_tiles(level0, coarse, 0, cancel_check=cancel_check)
    assert len(calls) == 3


def test_tissue_tile_contrast_falls_back_to_coarse_level_when_level0_read_fails(caplog):
    _, coarse, _ = _bright_tail_pyramid()

    class Broken:
        shape = (2048, 2048)
        data = property(lambda self: (_ for _ in ()).throw(OSError("chunk missing")))

    level0 = SimpleNamespace(isel=lambda c: Broken())
    with caplog.at_level("WARNING"):
        limits = contrast_limits_from_tissue_tiles(level0, coarse, 0)
    assert limits == contrast_limits_from_level(coarse, 0)
    assert "chunk missing" in caplog.text


class _LoggedPlane:
    """A 2-D plane whose ``.data`` records every level-0 slice read from it."""

    def __init__(self, array: np.ndarray, reads: list) -> None:
        self.shape = array.shape
        self._array = array
        self._reads = reads

    @property
    def data(self):
        return self

    def __getitem__(self, key):
        self._reads.append(key)
        return self._array[key]


def test_tissue_tiles_are_placed_on_off_origin_tissue_of_a_non_square_image():
    import xarray as xr

    height, width, block = 1024, 3072, 32
    full = np.zeros((1, height, width), dtype=np.uint16)
    # Tissue only in the bottom-right box, carrying a value found nowhere else.
    box_y, box_x = slice(640, 1024), slice(2048, 3072)
    full[0, box_y, box_x] = 777
    coarse = full.reshape(1, height // block, block, width // block, block).mean(axis=(2, 4))
    coarse_array = xr.DataArray(coarse, dims=("c", "y", "x"))
    reads: list = []
    plane = _LoggedPlane(full[0], reads)
    level0 = SimpleNamespace(isel=lambda c: plane)

    low, high = contrast_limits_from_tissue_tiles(level0, coarse_array, 0, n_tiles=16)

    assert (low, high) == (777.0, 778.0)
    assert len(reads) == 16
    for rows, cols in reads:
        # Every tile lies inside the image and reaches the tissue box on both axes.
        assert 0 <= rows.start < rows.stop <= height and 0 <= cols.start < cols.stop <= width
        assert rows.stop > box_y.start and cols.stop > box_x.start
        assert full[0, rows, cols].any()


def test_tissue_tile_contrast_reads_in_batches_and_polls_between_them():
    level0, coarse, _ = _bright_tail_pyramid()
    calls = []

    def cancel_check() -> bool:
        calls.append(1)
        return False

    contrast_limits_from_tissue_tiles(level0, coarse, 0, n_tiles=24, cancel_check=cancel_check)
    assert len(calls) == 3  # 24 tiles in batches of CONTRAST_TILE_BATCH (8)


def test_tissue_tiles_never_straddle_a_stored_chunk_boundary():
    import xarray as xr

    size, chunk, block = 1024, 256, 32
    full = np.full((1, size, size), 500, dtype=np.uint16)
    coarse = full.reshape(1, size // block, block, size // block, block).mean(axis=(2, 4))
    coarse_array = xr.DataArray(coarse, dims=("c", "y", "x"))
    reads: list = []
    plane = _LoggedPlane(full[0], reads)
    plane.chunks = ((chunk,) * (size // chunk), (chunk,) * (size // chunk))
    level0 = SimpleNamespace(isel=lambda c: plane)

    contrast_limits_from_tissue_tiles(level0, coarse_array, 0, n_tiles=32, tile_px=128)

    assert len(reads) == 32
    for rows, cols in reads:
        assert rows.start // chunk == (rows.stop - 1) // chunk
        assert cols.start // chunk == (cols.stop - 1) // chunk


def test_tissue_tiles_read_at_most_the_chunk_cap_of_distinct_chunks():
    import xarray as xr

    from napari_compare_xenium_merscope.manual_registration_controller import (
        CONTRAST_MAX_CHUNKS,
    )

    size, chunk, block = 2048, 256, 32
    rng = np.random.default_rng(1)
    full = rng.integers(50, 400, (1, size, size)).astype(np.uint16)
    coarse = full.reshape(1, size // block, block, size // block, block).mean(axis=(2, 4))
    coarse_array = xr.DataArray(coarse, dims=("c", "y", "x"))
    reads: list = []
    plane = _LoggedPlane(full[0], reads)
    plane.chunks = ((chunk,) * (size // chunk), (chunk,) * (size // chunk))
    level0 = SimpleNamespace(isel=lambda c: plane)

    contrast_limits_from_tissue_tiles(level0, coarse_array, 0, n_tiles=32, tile_px=128)

    assert len(reads) == 32
    touched = {(rows.start // chunk, cols.start // chunk) for rows, cols in reads}
    assert 1 < len(touched) <= CONTRAST_MAX_CHUNKS
    per_chunk = [
        sum(1 for rows, cols in reads if (rows.start // chunk, cols.start // chunk) == key)
        for key in touched
    ]
    assert max(per_chunk) - min(per_chunk) <= 1


def test_tissue_tiles_prefer_the_chunks_with_the_most_tissue():
    import xarray as xr

    size, chunk, block = 2048, 256, 32
    full = np.zeros((1, size, size), dtype=np.uint16)
    # Eight fully covered chunks plus one lightly covered chunk elsewhere.
    for index in range(8):
        full[0, 0:chunk, index * chunk : (index + 1) * chunk] = 500
    full[0, 1024 : 1024 + 64, 1024 : 1024 + 64] = 500
    coarse = full.reshape(1, size // block, block, size // block, block).mean(axis=(2, 4))
    coarse_array = xr.DataArray(coarse, dims=("c", "y", "x"))
    reads: list = []
    plane = _LoggedPlane(full[0], reads)
    plane.chunks = ((chunk,) * (size // chunk), (chunk,) * (size // chunk))
    level0 = SimpleNamespace(isel=lambda c: plane)

    contrast_limits_from_tissue_tiles(level0, coarse_array, 0, n_tiles=32, tile_px=128)

    assert {(rows.start // chunk, cols.start // chunk) for rows, cols in reads} == {
        (0, col) for col in range(8)
    }


def test_tissue_tiles_without_chunks_are_treated_as_one_chunk():
    import xarray as xr

    full = np.full((1, 1024, 1024), 300, dtype=np.uint16)
    coarse_array = xr.DataArray(np.full((1, 32, 32), 300.0), dims=("c", "y", "x"))
    level0 = xr.DataArray(full, dims=("c", "y", "x"))
    assert contrast_limits_from_tissue_tiles(level0, coarse_array, 0, n_tiles=8, tile_px=128) == (
        300.0,
        301.0,
    )
