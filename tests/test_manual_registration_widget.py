"""Tests for the Manual registration widget (offscreen Qt, plain pytest)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from qtpy.QtCore import QItemSelectionModel

from napari_compare_xenium_merscope.manual_registration_model import (
    CoarseState,
    CropRect,
    FitSummary,
    ResidualRow,
    finish_allowed,
)
from napari_compare_xenium_merscope.manual_registration_widget import (
    SIGNAL_NAMES,
    ManualRegistrationWidget,
)


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("qtpy/Qt not available")
    return QApplication.instance() or QApplication([])


class Recorder:
    def __init__(self, widget):
        self.calls = []
        for name in SIGNAL_NAMES:
            getattr(widget, name).connect(lambda *a, n=name: self.calls.append((n, a)))

    def pop(self):
        calls, self.calls = self.calls, []
        return calls


@pytest.fixture
def widget(qapp):
    w = ManualRegistrationWidget()
    yield w
    w.deleteLater()


def make_rows():
    return [
        ResidualRow(3, "#ff0000", True, True, False, False, 0.5, 0.7),
        ResidualRow(5, "#00ff00", True, False, True, True, 42.0, 41.0),
        ResidualRow(9, "#0000ff", False, False, False, False, float("nan"), float("nan")),
    ]


def make_summary(warnings=()):
    return FitSummary(
        n_enabled=10,
        n_used=9,
        scale=1.0,
        expected_scale=1.0,
        scale_ratio=1.00001339,
        rotation_deg=178.99913,
        reflected=False,
        rms_um=0.7,
        other_class_median_um=8000.0,
        chosen_class_median_um=0.6,
        spread_ratio=0.63,
        warnings=tuple(warnings),
    )


def test_setters_never_emit(widget):
    rec = Recorder(widget)
    w = widget
    w.set_session("prepared", "PAIR1")
    w.set_coarse(CoarseState(30, True, (1.5, -2.5)))
    w.set_coarse_enabled(False)
    w.set_display_mode("moving")
    w.set_landmark_mode(True)
    w.set_pending("moving", 4)
    w.set_rows(make_rows())
    w.set_fit_summary(make_summary(["a warning"]))
    w.set_crop("reference", CropRect(10, 20, 2000, 1500), (2000, 3000), 2)
    w.set_crop("moving", None, (2000, 3000), 0)
    w.set_crop_drawing("reference")
    w.set_preview_available(True)
    w.set_preview(True)
    w.set_finish_state(True, "")
    w.set_dirty(True, "3 pairs · draft")
    w.set_busy(True)
    w.set_busy(False)
    w.set_outlier_floor(12.0)
    w.set_session("result", "PAIR1_serial_section")
    w.set_result(make_rows(), "banner")
    w.set_overlay_visibility("landmark_fit_positions", True)
    w.set_session(None, "")
    assert rec.calls == []

    w.set_session("prepared", "PAIR1")
    w.set_display_mode("moving")
    w.set_landmark_mode(True)
    w.set_preview_available(True)
    w.set_preview(True)
    assert w._display_buttons["moving"].isChecked()
    assert w._landmark_button.isChecked() and w._landmark_button.text() == "Landmark mode: on"
    assert w._preview_button.isChecked()
    assert w.display_mode() == "moving" and w.landmark_mode() and w.preview()
    w.set_outlier_floor(12.0)
    assert w.outlier_floor_um() == 12.0
    w.set_coarse(CoarseState(30, True, (1.5, -2.5)))
    assert w._mirror_label.text() == "Mirrored: yes"
    assert w._rotation_spin.value() == 30.0
    assert (w._offset_x_spin.value(), w._offset_y_spin.value()) == (1.5, -2.5)
    assert rec.calls == []
    with pytest.raises(ValueError):
        w.set_display_mode("sideways")


def test_payloads(widget):
    w, rec = widget, Recorder(widget)
    w.set_session("prepared", "PAIR1")

    w._rotate_buttons[10].click()
    w._rotate_buttons[-0.1].click()
    w._rotate_180_button.click()
    assert rec.pop() == [("rotate_by", (10.0,)), ("rotate_by", (-0.1,)), ("rotate_by", (180.0,))]
    w._rotation_spin.setValue(-45.5)
    assert rec.pop() == [("rotation_set", (-45.5,))]

    w.set_coarse(CoarseState(0, False, (12.3456, -7.0)))
    w._offset_x_spin.setValue(20.0)
    assert rec.pop() == [("offset_set", (20.0, -7.0))]
    w._offset_y_spin.setValue(3.0)
    assert rec.pop() == [("offset_set", (12.3456, 3.0))]

    w._shift_step_combo.setCurrentIndex(1)
    assert w.shift_step_um() == 10.0
    for key in ("up", "left", "down", "right"):
        w._shift_buttons[key].click()
    assert rec.pop() == [
        ("shift_by", (0.0, -10.0)),
        ("shift_by", (-10.0, 0.0)),
        ("shift_by", (0.0, 10.0)),
        ("shift_by", (10.0, 0.0)),
    ]

    w.set_preview_available(True)
    w._flip_h_button.click()
    w._flip_v_button.click()
    w._reset_button.click()
    w._adopt_button.click()
    assert rec.pop() == [
        ("flip_horizontal", ()),
        ("flip_vertical", ()),
        ("reset_coarse", ()),
        ("adopt_fit", ()),
    ]

    w._display_buttons["reference"].click()
    w._flick_button.click()
    w._auto_switch_check.setChecked(False)
    w._landmark_button.click()
    assert rec.pop() == [
        ("display_mode_changed", ("reference",)),
        ("flick_requested", ()),
        ("auto_switch_changed", (False,)),
        ("landmark_mode_changed", (True,)),
    ]
    assert not w.auto_switch()

    w.set_rows([r for r in make_rows()])
    w._table.cellClicked.emit(1, 0)
    assert rec.pop() == [("pair_selected", (5,))]
    w._table.cellClicked.emit(1, 2)
    assert rec.pop() == []
    w._table.item(2, 2).setCheckState(2 if w._table.item(2, 2).checkState() == 0 else 0)
    assert rec.pop() == [("pair_enabled_changed", (9, True))]
    w._table.item(0, 2).setCheckState(0)
    assert rec.pop() == [("pair_enabled_changed", (3, False))]
    w._table.item(0, 2).setCheckState(2)
    rec.pop()
    w._table.selectRow(0)
    w._table.item(2, 0).setSelected(True)
    w._table.item(2, 1).setSelected(True)
    assert w._delete_button.isEnabled()
    w._delete_button.click()
    assert rec.pop() == [("delete_pairs", ([3, 9],))]

    w._undo_button.click()
    assert rec.pop() == [("undo_requested", ())]
    w._floor_spin.setValue(30.0)
    assert rec.pop() == [("outlier_floor_changed", (30.0,))]
    w._floor_preset_buttons["same_section"].click()
    assert rec.pop() == [("outlier_floor_changed", (10.0,))]
    w._floor_preset_buttons["serial_section"].click()
    assert rec.pop() == [("outlier_floor_changed", (30.0,))]
    w._floor_preset_buttons["serial_section"].click()  # unchanged: no emit
    assert rec.pop() == []
    w.set_outlier_floor(12.0)
    assert rec.pop() == []
    w._preview_button.click()
    assert rec.pop() == [("preview_changed", (True,))]

    w.set_crop("reference", None, (2000, 3000), 0)
    w._crop_draw_buttons["reference"].click()
    assert rec.pop() == [("crop_draw_requested", ("reference",))]
    w._crop_spins["reference"]["x0"].setValue(100)
    assert rec.pop() == [("crop_set", ("reference", CropRect(100, 0, 3000, 2000)))]
    w._crop_spins["reference"]["x0"].setValue(0)
    assert rec.pop() == [("crop_set", ("reference", None))]
    w._crop_clear_buttons["reference"].click()
    assert rec.pop() == [("crop_set", ("reference", None))]
    w._crop_spins["reference"]["x0"].setValue(100)
    rec.pop()
    w._crop_spins["reference"]["x1"].setValue(50)
    assert rec.pop() == []
    assert "Invalid crop" in w._crop_labels["reference"].text()
    # a rectangle under 16 px is invalid too
    w.set_crop("moving", None, (2000, 3000), 0)
    w._crop_spins["moving"]["x1"].setValue(10)
    assert rec.pop() == []
    assert "Invalid crop" in w._crop_labels["moving"].text()

    w._save_button.click()
    w.set_finish_state(True, "")
    w._finish_button.click()
    assert rec.pop() == [("save_requested", ()), ("finish_requested", ())]

    w.set_session("result", "PAIR1_serial_section")
    w._overlay_checks["registered_landmarks"].setChecked(False)
    assert rec.pop() == [("overlay_changed", ("registered_landmarks", False))]
    w.set_result(make_rows(), "banner")
    w._result_table.cellClicked.emit(0, 0)
    assert rec.pop() == [("pair_selected", (3,))]


def test_enable_logic(widget):
    w = widget
    w.set_session(None, "")
    assert not w._editing_box.isEnabled() and not w._save_button.isEnabled()
    assert "No registration store loaded" in w._status_label.text()

    w.set_session("prepared", "PAIR1")
    assert w._editing_box.isEnabled() and not w._editing_box.isHidden()
    assert w._result_box.isHidden()
    w.set_finish_state(*finish_allowed(7, False, False))
    assert not w._finish_button.isEnabled() and "8" in w._finish_button.toolTip()
    w.set_finish_state(*finish_allowed(8, True, False))
    assert not w._finish_button.isEnabled() and "pending" in w._finish_button.toolTip()
    w.set_finish_state(*finish_allowed(8, False, False))
    assert w._finish_button.isEnabled()
    w.set_busy(True)
    assert not w._save_button.isEnabled() and not w._finish_button.isEnabled()
    assert w._busy_label.text() == "Saving…"
    assert not w._crop_box.isEnabled() and not w._coarse_box.isEnabled()
    assert not w._landmark_box.isEnabled()
    w.set_busy(False)
    assert w._finish_button.isEnabled() and w._save_button.isEnabled()
    assert w._coarse_box.isEnabled() and w._busy_label.text() == ""

    assert not w._preview_button.isEnabled() and not w._adopt_button.isEnabled()
    w.set_preview_available(True)
    assert w._preview_button.isEnabled() and w._adopt_button.isEnabled()
    assert not w._preview_button.isChecked()

    w.set_coarse_enabled(False)
    for widget_ in (
        w._rotation_spin,
        w._flip_h_button,
        w._flip_v_button,
        w._reset_button,
        *w._shift_buttons.values(),
    ):
        assert not widget_.isEnabled()
    w.set_busy(True)
    w.set_busy(False)
    assert not w._rotation_spin.isEnabled()  # coarse state restored from the last setter
    w.set_coarse_enabled(True)
    assert w._rotation_spin.isEnabled()

    assert not w._delete_button.isEnabled()
    w.set_rows(make_rows())
    assert not w._delete_button.isEnabled()
    w._table.selectRow(1)
    assert w._delete_button.isEnabled()
    w.set_rows(make_rows())  # the selection is kept by id
    assert w._selected_ids() == [5]
    w._table.selectRow(0)
    w._table.selectionModel().select(
        w._table.model().index(2, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows
    )
    assert w._selected_ids() == [3, 9]
    w.set_rows(make_rows())  # a multi-row selection survives too
    assert w._selected_ids() == [3, 9]
    w._table.selectRow(1)
    w.set_rows([make_rows()[0]])
    assert not w._delete_button.isEnabled()

    w.set_crop_drawing("moving")
    assert w._crop_draw_buttons["moving"].text() == "Drawing… (Esc cancels)"
    assert not w._crop_draw_buttons["reference"].isEnabled()
    w.set_crop_drawing(None)
    assert w._crop_draw_buttons["moving"].text() == "Draw"
    assert w._crop_draw_buttons["reference"].isEnabled()

    w.set_session("result", "PAIR1_registered")
    assert w._editing_box.isHidden() and not w._result_box.isHidden()
    assert w._display_buttons["moving"].text() == "Registered moving"
    assert w._display_box.isEnabled()
    w._display_buttons["reference"].click()
    assert w._display_buttons["reference"].isChecked()
    w.set_session("prepared", "PAIR1")
    assert w._display_buttons["moving"].text() == "Moving"
    assert w._display_buttons["both"].isChecked()  # a kind change resets the display


def test_session_kind_change_resets_state(widget):
    w = widget
    w.set_session("prepared", "PAIR1")
    w.set_landmark_mode(True)
    w.set_preview_available(True)
    w.set_preview(True)
    w.set_display_mode("reference")
    w.set_rows(make_rows())
    w.set_fit_summary(make_summary(["w"]))
    w.set_session("result", "R")
    w.set_session("prepared", "PAIR1")
    assert not w.landmark_mode() and not w.preview()
    assert w.display_mode() == "both"
    assert w._table.rowCount() == 0
    assert "at least 2 enabled pairs" in w._summary_label.text()
    with pytest.raises(ValueError):
        w.set_session("bogus", "")


def test_labels(widget):
    w = widget
    w.set_session("prepared", "PAIR1")
    w.set_crop("reference", CropRect(1000, 500, 71000, 50500), (55012, 76544), 3)
    text = w._crop_labels["reference"].text()
    assert "70000 × 50000" in text and "effective 69952 × 49984" in text
    assert "3 pair(s) outside" in w._crop_outside_labels["reference"].text()
    assert (
        w._crop_spins["reference"]["x1"].value(),
        w._crop_spins["reference"]["x1"].maximum(),
    ) == (
        71000,
        76544,
    )
    w.set_crop("reference", CropRect(0, 0, 1000, 1000), (3000, 3000), 0)
    assert "too small" in w._crop_labels["reference"].text()
    assert w._crop_outside_labels["reference"].text() == ""
    w.set_crop("moving", None, (2000, 3000), 0)
    assert w._crop_labels["moving"].text() == "Full image (3000 × 2000 px)"
    spins = w._crop_spins["moving"]
    assert [spins[k].value() for k in ("x0", "y0", "x1", "y1")] == [0, 0, 3000, 2000]

    w.set_fit_summary(None)
    assert "at least 2 enabled pairs" in w._summary_label.text()
    w.set_fit_summary(make_summary(["first", "second"]))
    assert w._warnings_label.text() == "first\nsecond"
    assert "Fit on 9 of 10 enabled pairs" in w._summary_label.text()
    assert "mirrored no" in w._summary_label.text()

    w.set_landmark_mode(False)
    w.set_pending("reference", 4)
    assert "REFERENCE" in w._prompt_label.text() and "pair 4" in w._prompt_label.text()
    w.set_pending("moving", 4)
    assert "MOVING" in w._prompt_label.text()
    w.set_pending(None, None)
    assert w._prompt_label.text() == ""
    w.set_landmark_mode(True)
    assert w._prompt_label.text() == "Landmark mode is on"
    w.set_dirty(True, "12 pairs · finished · saved")
    assert w._status_label.text() == "● unsaved changes · 12 pairs · finished · saved"
    w.set_dirty(False, "12 pairs")
    assert w._status_label.text() == "12 pairs"


def test_table_content(widget):
    w = widget
    w.set_session("prepared", "PAIR1")
    w.set_rows(make_rows())
    t = w._table
    assert [t.item(r, 0).text() for r in range(3)] == ["3", "5", "9"]
    assert t.item(0, 3).text() == "0.50" and t.item(2, 3).text() == "–"
    assert t.item(1, 5).text() == "⚠ outlier ✂ outside crop"
    assert t.item(2, 5).text() == "off"
    assert t.item(0, 1).background().color().name() == "#ff0000"
    assert t.item(0, 2).checkState() == 2 and t.item(2, 2).checkState() == 0
    w.set_session("result", "R")
    rows = [ResidualRow(1, "#123456", False, False, False, False, 1.0, 2.0, 3.0, None)]
    w.set_result(rows, "ok")
    rt = w._result_table
    assert rt.item(0, 4).text() == "3.00" and rt.item(0, 5).text() == "–"
    assert rt.item(0, 6).text() == "held out"
    assert w._result_banner.text() == "ok"
    assert [k for k, c in w._overlay_checks.items() if c.isChecked()] == [
        "reference_landmarks",
        "registered_landmarks",
    ]
    assert w._shift_step_combo.currentData() == 100.0
    assert [w._shift_step_combo.itemText(i) for i in range(4)] == [
        "1 µm",
        "10 µm",
        "100 µm",
        "1000 µm",
    ]


def test_rotation_spin_arrows_cross_180(qapp):
    w = ManualRegistrationWidget()
    rec = Recorder(w)
    w.set_coarse(CoarseState(180.0, False, (0.0, 0.0)))
    w.set_coarse_enabled(True)
    rec.pop()
    w._rotation_spin.stepBy(1)
    assert rec.pop() == [("rotation_set", (-179.0,))]
    w._rotation_spin.setValue(-179.0)
    rec.pop()
    w._rotation_spin.stepBy(-1)
    assert rec.pop() == [("rotation_set", (180.0,))]
