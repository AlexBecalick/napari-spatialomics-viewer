"""Tests for the pure manual-registration model (no Qt, no zarr, no napari)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from napari_compare_xenium_merscope import manual_registration_model as M
from napari_compare_xenium_merscope.manual_registration_model import (
    LABELLING_ATTR,
    STORE_ATTR,
    CoarseState,
    CropRect,
    LabellingMismatch,
    LabellingState,
    LandmarkPair,
    NotARegistrationStore,
    RobustFitOptions,
    UnsupportedSchemaVersion,
    apply_matrix_xy,
    canonical_json,
    canonical_sha256,
    coarse_from_matrix,
    coarse_matrix_xy,
    evaluate_landmarks,
    finish_allowed,
    fit_similarity,
    labelling_content_sha256,
    labelling_from_attrs,
    labelling_to_attrs,
    leave_one_out_residuals,
    napari_affine_rc,
    pair_colour,
    pairs_outside_crop,
    parse_store_descriptor,
    points_outside_crop,
    robust_similarity,
    rotation_matrix_xy,
    snap_crop,
    spread_ratio,
    translation_xy,
    umeyama_similarity,
    wrap_deg,
)

EXAMPLE_SHA = "sha256:f6b16674a10854ab06ff385a6a349ce325d874abf0e6c9d78e0a3b7683e328ad"

EXAMPLE_LABELLING = {
    "schema": "merxen.manual_registration.labelling",
    "schema_version": 1,
    "status": "finished",
    "revision": 9,
    "updated_at": "2026-11-02T14:02:11Z",
    "finished_at": "2026-11-02T14:02:11Z",
    "written_by": {"software": "napari-compare-xenium-merscope", "version": "0.4.0"},
    "store_grid_digest": "sha256:" + "0" * 64,
    "images": {
        "reference": {"shape_yx": [55012, 76544], "pixel_size_um": 0.34511},
        "moving": {"shape_yx": [61003, 80112], "pixel_size_um": 0.34511},
    },
    "coarse": {
        "rotation_deg": 179.2,
        "mirror_x": False,
        "offset_um": [152.4, -88.0],
        "matrix_xy": [
            [-0.9999025, 0.0139622, 78338.8334],
            [-0.0139622, -0.9999025, 58307.7978],
            [0.0, 0.0, 1.0],
        ],
    },
    "crops": {"reference": {"x0": 1000, "y0": 500, "x1": 71000, "y1": 50500}, "moving": None},
    "landmarks": {
        "next_id": 13,
        "colour_seed": 4211,
        "pairs": [
            {
                "id": 1,
                "colour": "#fff400",
                "enabled": True,
                "reference_xy_px": [12034.5, 8811.0],
                "moving_xy_px": [64310.25, 49702.75],
            },
            {
                "id": 2,
                "colour": "#b219ff",
                "enabled": False,
                "reference_xy_px": [40211.0, 30122.5],
                "moving_xy_px": [36420.0, 28533.0],
            },
        ],
    },
    "content_sha256": EXAMPLE_SHA,
}


def prepared_store_attr(
    ref_shape=(55012, 76544), mov_shape=(61003, 80112), p_r=0.34511, p_m=0.34511
):
    def rec(role, shape, px):
        return {
            "element": role,
            "role": role,
            "shape_yx": list(shape),
            "dtype": "uint16",
            "pixel_size_um": px,
            "channel_names": ["DAPI", "AF568", "AF647"],
            "registration_channel": "DAPI",
        }

    return {
        "schema": "merxen.manual_registration.store",
        "schema_version": 1,
        "kind": "prepared",
        "dataset_name": "PAIR1",
        "images": {
            "reference": rec("reference", ref_shape, p_r),
            "moving": rec("moving", mov_shape, p_m),
        },
        "grid_digest": "sha256:" + "a" * 64,
        "content_digest": "sha256:" + "b" * 64,
    }


def result_store_attr():
    px = 0.34511
    chans = ["DAPI", "AF568", "AF647"]
    return {
        "schema": "merxen.manual_registration.store",
        "schema_version": 1,
        "kind": "result",
        "dataset_name": "PAIR1",
        "run_id": "PAIR1_serial_section",
        "preset": "serial_section",
        "run_label": None,
        "prepared_store": {
            "path": "PAIR1_prepared.zarr",
            "content_digest": "sha256:" + "b" * 64,
            "grid_digest": "sha256:" + "a" * 64,
        },
        "images": {
            "reference": {
                "element": "reference",
                "role": "reference",
                "shape_yx": [49984, 69952],
                "full_shape_yx": [55012, 76544],
                "crop": {"x0": 1024, "y0": 512, "x1": 70976, "y1": 50496},
                "pixel_size_um": px,
                "dtype": "uint16",
                "channel_names": chans,
                "registration_channel": "DAPI",
            },
            "moving": {
                "element": "moving_registered",
                "role": "moving",
                "shape_yx": [49984, 69952],
                "pixel_size_um": px,
                "dtype": "uint16",
                "channel_names": chans,
                "registration_channel": "DAPI",
                "source_shape_yx": [61003, 80112],
                "source_pixel_size_um": px,
                "source_crop": {"x0": 0, "y0": 0, "x1": 80112, "y1": 61003},
            },
        },
    }


RESULT_ATTR_MIN = {
    "schema": "merxen.manual_registration.result",
    "schema_version": 1,
    "status": "complete",
}


def prepared_descriptor(**kw):
    return parse_store_descriptor(
        Path("PAIR1_prepared.zarr"), {STORE_ATTR: prepared_store_attr(**kw)}
    )


# ---------------------------------------------------------------------------------------------
# Appendix A.1 and coarse properties
# ---------------------------------------------------------------------------------------------

A1_CASES = [
    (0, False, (0, 0), (1000, 2000), (1000, 2000), 0.5, 0.5, [[1, 0, 0], [0, 1, 0]]),
    (
        30,
        False,
        (12.5, -40),
        (1001, 1500),
        (2000, 3001),
        0.3434,
        0.2125,
        [[1.3994970525, 0.808, 105.9004885513], [-0.808, 1.3994970525, 717.1121796245]],
    ),
    (
        -125,
        True,
        (-100, 250),
        (48000, 72000),
        (61000, 80000),
        0.34511,
        0.34511,
        [
            [0.5735764364, -0.8191520443, 38720.5118821327],
            [-0.8191520443, -0.5735764364, 74478.5184134923],
        ],
    ),
    (
        178.5,
        False,
        (120, -35.5),
        (31347, 33453),
        (51313, 47618),
        0.3434,
        0.2125,
        [
            [-1.6154462372, 0.0423019485, 50730.1612067993],
            [-0.0423019485, -1.6154462372, 51515.3724415215],
        ],
    ),
]


@pytest.mark.parametrize("theta,mirror,offset,mov,ref,p_m,p_r,expected", A1_CASES)
def test_appendix_a1_coarse_matrices(theta, mirror, offset, mov, ref, p_m, p_r, expected):
    m = coarse_matrix_xy(
        CoarseState(theta, mirror, offset),
        moving_shape_yx=mov,
        reference_shape_yx=ref,
        moving_pixel_size_um=p_m,
        reference_pixel_size_um=p_r,
    )
    exp = np.array(expected, dtype=float)
    assert np.allclose(m[:2, :2], exp[:, :2], atol=1e-9)
    assert np.allclose(m[:2, 2], exp[:, 2], atol=1e-6)
    assert np.allclose(m[2], [0, 0, 1])


def test_example_matrix_of_section_3_6():
    m = coarse_matrix_xy(
        CoarseState(179.2, False, (152.4, -88.0)),
        moving_shape_yx=(61003, 80112),
        reference_shape_yx=(55012, 76544),
        moving_pixel_size_um=0.34511,
        reference_pixel_size_um=0.34511,
    )
    exp = np.array(EXAMPLE_LABELLING["coarse"]["matrix_xy"])
    assert np.allclose(m, exp, atol=1e-6)
    assert np.allclose(
        m[:2],
        [[-0.9999025, 0.0139622, 78338.8334189], [-0.0139622, -0.9999025, 58307.7978371]],
        atol=1e-6,
    )


MOV, REF, P_M, P_R = (31347, 33453), (51313, 47618), 0.3434, 0.2125
OFFSET = (120.0, -35.5)
CASE_KW = dict(
    moving_shape_yx=MOV,
    reference_shape_yx=REF,
    moving_pixel_size_um=P_M,
    reference_pixel_size_um=P_R,
)
THETAS = [0.0, 30.0, -125.0, 178.5, 180.0]
COMBOS = [(t, m) for t in THETAS for m in (False, True)]


def _matrix(state):
    return coarse_matrix_xy(state, **CASE_KW)


def _displayed_centre(state):
    cr = M.image_centre_xy(REF)
    return (cr[0] + state.offset_um[0] / P_R, cr[1] + state.offset_um[1] / P_R)


@pytest.mark.parametrize("theta,mirror", COMBOS)
def test_coarse_properties(theta, mirror):
    state = CoarseState(theta, mirror, OFFSET)
    m = _matrix(state)
    cm = np.array(M.image_centre_xy(MOV))
    cr = np.array(M.image_centre_xy(REF))
    assert np.allclose(apply_matrix_xy(m, cm), cr + np.array(OFFSET) / P_R, atol=1e-6)
    assert (np.linalg.det(m[:2, :2]) < 0) == mirror
    d = _displayed_centre(state)
    t_d, t_nd = translation_xy(*d), translation_xy(-d[0], -d[1])

    new = state.rotated(37.25)
    assert np.allclose(
        _matrix(new), t_d @ rotation_matrix_xy(37.25) @ t_nd @ m, rtol=1e-12, atol=1e-6
    )
    new = state.flipped_horizontal()
    assert np.allclose(_matrix(new), t_d @ np.diag([-1.0, 1, 1]) @ t_nd @ m, rtol=1e-12, atol=1e-6)
    assert new.mirror_x == (not mirror)
    new = state.flipped_vertical()
    assert np.allclose(_matrix(new), t_d @ np.diag([1.0, -1, 1]) @ t_nd @ m, rtol=1e-12, atol=1e-6)
    new = state.shifted(12.5, -3.0)
    assert np.allclose(
        _matrix(new), translation_xy(12.5 / P_R, -3.0 / P_R) @ m, rtol=1e-12, atol=1e-6
    )
    # inverse
    back, ratio = coarse_from_matrix(m, **CASE_KW)
    assert abs(wrap_deg(back.rotation_deg - state.rotation_deg)) < 1e-9
    assert back.mirror_x == state.mirror_x
    assert np.allclose(back.offset_um, state.offset_um, atol=1e-6)
    assert ratio == pytest.approx(1.0, abs=1e-12)
    # scaled matrix reports its scale ratio
    _, ratio = coarse_from_matrix(m @ np.diag([1.1, 1.1, 1.0]), **CASE_KW)
    assert ratio == pytest.approx(1.1, abs=1e-9)


def test_reset_is_identity_for_equal_shapes():
    m = coarse_matrix_xy(
        CoarseState(),
        moving_shape_yx=(1000, 2000),
        reference_shape_yx=(1000, 2000),
        moving_pixel_size_um=0.5,
        reference_pixel_size_um=0.5,
    )
    assert np.allclose(m, np.eye(3))


def test_coarse_state_helpers_and_validation():
    s = CoarseState(rotation_deg=-180)
    assert s.rotation_deg == 180.0
    assert CoarseState(10).with_rotation(190).rotation_deg == -170.0
    assert CoarseState().with_offset(1, 2).offset_um == (1.0, 2.0)
    assert CoarseState(170).rotated(20).rotation_deg == -170.0
    with pytest.raises(ValueError):
        coarse_matrix_xy(
            CoarseState(),
            moving_shape_yx=(10, 10),
            reference_shape_yx=(10, 10),
            moving_pixel_size_um=0.0,
            reference_pixel_size_um=1.0,
        )
    with pytest.raises(ValueError):
        coarse_matrix_xy(
            CoarseState(),
            moving_shape_yx=(0, 10),
            reference_shape_yx=(10, 10),
            moving_pixel_size_um=1.0,
            reference_pixel_size_um=1.0,
        )


def test_wrap_deg_values():
    assert wrap_deg(180) == 180
    assert wrap_deg(-180) == 180
    assert wrap_deg(190) == -170
    assert wrap_deg(-190) == 170
    assert wrap_deg(540) == 180
    with pytest.raises(ValueError):
        wrap_deg(float("nan"))


def test_napari_affine_rc():
    a_ref = np.diag([0.2125, 0.2125, 1.0])
    _, mirror, offset, mov, ref, p_m, p_r, _ = A1_CASES[1]
    m = coarse_matrix_xy(
        CoarseState(30, mirror, offset),
        moving_shape_yx=mov,
        reference_shape_yx=ref,
        moving_pixel_size_um=p_m,
        reference_pixel_size_um=p_r,
    )
    a_mov = napari_affine_rc(a_ref, m)
    mx, my = 321.5, 77.25
    got = a_mov @ np.array([my, mx, 1.0])
    ref_pt = apply_matrix_xy(m, np.array([mx, my]))
    assert np.allclose(got[:2], p_r * np.array([ref_pt[1], ref_pt[0]]), atol=1e-9)
    assert M.SWAP.flags.writeable is False


# ---------------------------------------------------------------------------------------------
# Crops
# ---------------------------------------------------------------------------------------------


def test_crop_containment_and_validation():
    crop = CropRect(10, 20, 110, 220)
    inside = np.array([(9.5, 19.5), (109.4999, 219.4999)])
    outside = np.array([(9.4999, 50), (109.5, 50), (50, 19.4999), (50, 219.5)])
    assert crop.contains_xy(inside).all()
    assert not crop.contains_xy(outside).any()
    assert not points_outside_crop(inside, None).any()
    assert points_outside_crop(outside, crop).all()
    for bad in [(5, 0, 5, 10), (-1, 0, 5, 5), (0.5, 0, 5, 5), (True, 0, 5, 5)]:
        with pytest.raises(ValueError):
            CropRect(*bad)
    with pytest.raises(ValueError):
        CropRect(0, 0, 5, 5).validate_within((10, 4))
    CropRect(0, 0, 4, 5).validate_within((10, 4))
    assert CropRect(1, 2, 5, 9).width == 4 and CropRect(1, 2, 5, 9).height == 7
    assert CropRect.from_attrs(None) is None
    assert CropRect.from_attrs(CropRect(1, 2, 5, 9).to_attrs()) == CropRect(1, 2, 5, 9)
    with pytest.raises(ValueError):
        CropRect.from_attrs({"x0": 1})


def test_pairs_outside_crop_moving_only():
    pairs = [
        LandmarkPair(1, "#000000", (50, 50), (50, 50)),
        LandmarkPair(2, "#000000", (50, 50), (500, 50)),
    ]
    crops = {"reference": CropRect(0, 0, 100, 100), "moving": CropRect(0, 0, 100, 100)}
    assert pairs_outside_crop(pairs, crops).tolist() == [False, True]
    assert pairs_outside_crop(pairs, {"reference": None, "moving": None}).tolist() == [False, False]
    assert pairs_outside_crop([], crops).shape == (0,)


def test_point_in_image():
    assert M.point_in_image((-0.5, -0.5), (10, 20))
    assert not M.point_in_image((19.5, 0), (10, 20))
    assert M.point_in_image((19.4999, 9.4999), (10, 20))


def test_snap_crop_cases():
    assert snap_crop(CropRect(1000, 500, 71000, 50500), (55012, 76544)) == CropRect(
        1024, 512, 70976, 50496
    )
    assert snap_crop(CropRect(100, 100, 76544, 55012), (55012, 76544)) == CropRect(
        128, 128, 76544, 55012
    )
    assert snap_crop(None, (2000, 3000)) == CropRect(0, 0, 3000, 2000)
    assert snap_crop(CropRect(0, 0, 1050, 3000), (3000, 3000)) == CropRect(0, 0, 1024, 3000)
    with pytest.raises(ValueError, match="960"):
        snap_crop(CropRect(0, 0, 1000, 3000), (3000, 3000))
    with pytest.raises(ValueError):
        snap_crop(CropRect(0, 0, 10, 10), (5, 5))


# ---------------------------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [0.0, 73.0, 180.0, -125.0])
@pytest.mark.parametrize("scale", [0.6, 1.0, 1.6])
@pytest.mark.parametrize("refl", [False, True])
def test_umeyama_recovers_truth(theta, scale, refl):
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 5000, (20, 2))
    truth = (
        translation_xy(800, -300)
        @ rotation_matrix_xy(theta)
        @ np.diag([-1.0 if refl else 1.0, 1.0, 1.0])
        @ np.diag([scale, scale, 1.0])
    )
    dst = apply_matrix_xy(truth, src) + rng.normal(0, 0.5, src.shape)
    fit = fit_similarity(src, dst, reflection=refl)
    assert np.allclose(fit.matrix[:2, :2], truth[:2, :2], atol=1e-3)
    assert np.allclose(fit.matrix[:2, 2], truth[:2, 2], atol=2.0)
    assert fit.reflected == refl
    assert abs(wrap_deg(fit.rotation_deg - theta)) < 0.05
    assert np.allclose(umeyama_similarity(src, dst, reflection=refl), fit.matrix)
    wrong = fit_similarity(src, dst, reflection=not refl)
    assert np.median(wrong.residuals_px) > 50 * np.median(fit.residuals_px)


def test_fit_degenerate_and_small_inputs():
    pts = np.tile([[5.0, 5.0]], (4, 1))
    with pytest.raises(ValueError):
        umeyama_similarity(pts, pts + 1, reflection=False)
    with pytest.raises(ValueError, match="at least 2"):
        fit_similarity([[1, 2]], [[3, 4]], reflection=False)
    with pytest.raises(ValueError):
        fit_similarity([[1, 2], [3, 4]], [[1, 2]], reflection=False)
    with pytest.raises(ValueError):
        fit_similarity([[1, 2], [3, np.nan]], [[1, 2], [3, 4]], reflection=False)
    two = leave_one_out_residuals([[0, 0], [1, 1]], [[0, 0], [1, 1]], reflection=False)
    assert np.isnan(two).all()


def test_spread_ratio_values():
    assert spread_ratio([[0, 0], [1, 0], [0, 1], [1, 1]]) == pytest.approx(1.0, abs=1e-9)
    assert spread_ratio([[0, 0], [1, 1], [2, 2]]) == pytest.approx(0.0, abs=1e-9)
    assert spread_ratio([[0, 0]]) == 0.0


A2_MOVING = [
    (5000, 4000),
    (30000, 6000),
    (65000, 9000),
    (12000, 24000),
    (40000, 22000),
    (68000, 30000),
    (8000, 42000),
    (36000, 44000),
    (60000, 40000),
    (20000, 12000),
]
A2_REFERENCE = [
    (71081.083, 50804.36),
    (46116.596, 48370.654),
    (11176.784, 44760.977),
    (64428.598, 30684.939),
    (36517.762, 32197.367),
    (8542.641, 23707.218),
    (68745.332, 12759.59),
    (40781.701, 10270.827),
    (16716.747, 13849.06),
    (56219.587, 42546.592),
]
P = 0.34511


def _true_matrix():
    return coarse_matrix_xy(
        CoarseState(179, False, (150, -80)),
        moving_shape_yx=(48000, 72000),
        reference_shape_yx=(61000, 80000),
        moving_pixel_size_um=P,
        reference_pixel_size_um=P,
    )


def test_appendix_a2_robust_example():
    true = _true_matrix()
    assert np.allclose(
        true[:2],
        [[-0.999848, 0.017452, 76009.3121], [-0.017452, -0.999848, 54891.812518]],
        atol=1e-4,
    )
    src, dst = np.array(A2_MOVING, float), np.array(A2_REFERENCE, float)
    r = robust_similarity(
        src,
        dst,
        reflection=False,
        moving_pixel_size_um=P,
        reference_pixel_size_um=P,
        options=RobustFitOptions(),
    )
    expected_used = np.ones(10, bool)
    expected_used[4] = False
    assert r.used.tolist() == expected_used.tolist()
    assert r.outlier.tolist() == (~expected_used).tolist()
    m = r.fit.matrix
    assert np.allclose(m[:2, :2], [[-0.999861, 0.017468], [-0.017468, -0.999861]], atol=1e-5)
    assert np.allclose(m[:2, 2], [76009.005205, 54892.785194], atol=1e-3)
    assert r.fit.rotation_deg == pytest.approx(178.99913, abs=1e-4)
    assert r.fit.scale == pytest.approx(1.00001339, abs=1e-7)
    res = [0.7702, 0.6055, 0.9654, 0.7053, 41.0328, 0.668, 0.5918, 0.4061, 0.4827, 0.8113]
    loo = [1.0707, 0.7173, 1.3361, 0.8578, 41.0328, 0.9301, 0.7922, 0.49, 0.6404, 0.9614]
    first = [3.8451, 5.459, 6.4563, 4.6403, 41.0328, 7.0818, 3.4237, 5.1563, 6.1226, 5.2289]
    assert np.allclose(r.residuals_um, res, atol=1e-3)
    assert np.allclose(r.loo_um, loo, atol=1e-3)
    assert np.allclose(leave_one_out_residuals(src, dst, reflection=False) * P, first, atol=1e-3)
    assert r.chosen_class_median_um == pytest.approx(0.6680, abs=1e-3)
    assert r.other_class_median_um == pytest.approx(8278.3833, abs=1e-3)
    assert r.spread_ratio == pytest.approx(0.62677, abs=1e-4)
    # all-pairs medians
    proper = fit_similarity(src, dst, reflection=False)
    refl = fit_similarity(src, dst, reflection=True)
    assert np.median(proper.residuals_px * P) == pytest.approx(4.5655, abs=1e-3)
    assert np.median(refl.residuals_px * P) == pytest.approx(8131.7391, abs=1e-3)


TR = coarse_matrix_xy(
    CoarseState(30),
    moving_shape_yx=(4000, 4000),
    reference_shape_yx=(4000, 4000),
    moving_pixel_size_um=1.0,
    reference_pixel_size_um=1.0,
)


def _mk(n, seed):
    g = np.random.default_rng(seed)
    m = g.uniform(0, 4000, (n, 2))
    r = apply_matrix_xy(TR, m) + g.normal(0, 0.5, (n, 2))
    return m, r


def _robust(m, r, **opts):
    return robust_similarity(
        m,
        r,
        reflection=False,
        moving_pixel_size_um=1.0,
        reference_pixel_size_um=1.0,
        options=RobustFitOptions(floor_um=10.0, k=3.0, **opts),
    )


def _excluded(result):
    return set(np.flatnonzero(~result.used).tolist())


def test_trimming_stops_at_fraction_and_min_used():
    m, r = _mk(10, 2)
    r[1, 0] += 200
    r[6, 1] -= 300
    assert _excluded(_robust(m, r)) == {1, 6}
    assert _excluded(_robust(m, r, max_outlier_fraction=0.1)) == {6}

    m, r = _mk(7, 3)
    r[1, 0] += 200
    r[5, 1] -= 300
    assert _excluded(_robust(m, r, max_outlier_fraction=0.49, min_used_pairs=6)) == {5}
    assert _excluded(_robust(m, r, max_outlier_fraction=0.49, min_used_pairs=5)) == {1, 5}

    m, r = _mk(10, 2)
    r[4, 0] += 30
    assert _excluded(_robust(m, r)) == {4}


def test_trim_outliers_false_flags_without_excluding():
    m, r = _mk(10, 2)
    r[1, 0] += 200
    r[6, 1] -= 300
    res = _robust(m, r, trim_outliers=False)
    assert res.used.all()
    loo_all = leave_one_out_residuals(m, r, reflection=False)
    med = np.median(loo_all)
    assert res.outlier.tolist() == ((loo_all > 10.0) & (loo_all > 3.0 * med)).tolist()
    assert res.outlier.any()


def test_robust_rejects_bad_input():
    with pytest.raises(ValueError):
        robust_similarity(
            [[0, 0]],
            [[0, 0]],
            reflection=False,
            moving_pixel_size_um=1.0,
            reference_pixel_size_um=1.0,
        )
    with pytest.raises(ValueError):
        robust_similarity(
            [[0, 0], [1, 1]],
            [[0, 0], [1, 1]],
            reflection=False,
            moving_pixel_size_um=0.0,
            reference_pixel_size_um=1.0,
        )


# ---------------------------------------------------------------------------------------------
# Colours, digests, attrs
# ---------------------------------------------------------------------------------------------


def test_pair_colours():
    seed_4211 = (
        "#fff400 #b219ff #40ff87 #ff1400 #194aff #a0ff40 #ff00cb #19ffeb #ffb640 #5400ff "
        "#19ff22 #ff4070"
    ).split()
    seed_0 = (
        "#004aff #9fff19 #ff40e7 #00ffd4 #ff9619 #6f40ff #0bff00 #ff1966 #40b8ff #eaff00 "
        "#cf19ff #40ff9f"
    ).split()
    assert [pair_colour(i, 4211) for i in range(1, 13)] == seed_4211
    assert [pair_colour(i, 0) for i in range(1, 13)] == seed_0


def test_digests():
    obj = {"a": 1, "b": [1.5, "x", None, True]}
    assert (
        canonical_sha256(obj)
        == "sha256:19382923a583ec2cc1069a29913e5695ea02f5991790b1b9feec0158962c8a5d"
    )
    assert labelling_content_sha256(EXAMPLE_LABELLING) == EXAMPLE_SHA
    again = json.loads(json.dumps(EXAMPLE_LABELLING))
    assert labelling_content_sha256(again) == EXAMPLE_SHA
    tupled = copy.deepcopy(EXAMPLE_LABELLING)
    tupled["landmarks"]["pairs"] = tuple(
        {
            **p,
            "reference_xy_px": tuple(p["reference_xy_px"]),
            "moving_xy_px": tuple(p["moving_xy_px"]),
        }
        for p in tupled["landmarks"]["pairs"]
    )
    tupled["coarse"]["offset_um"] = tuple(tupled["coarse"]["offset_um"])
    assert labelling_content_sha256(tupled) == EXAMPLE_SHA
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_attrs_round_trip():
    d = parse_store_descriptor(
        Path("PAIR1_prepared.zarr"),
        {STORE_ATTR: prepared_store_attr(), LABELLING_ATTR: EXAMPLE_LABELLING},
    )
    s = labelling_from_attrs(EXAMPLE_LABELLING, d)
    assert s.coarse.rotation_deg == 179.2
    assert s.coarse.offset_um == (152.4, -88.0)
    assert s.coarse.mirror_x is False
    assert s.crops["reference"] == CropRect(1000, 500, 71000, 50500)
    assert s.crops["moving"] is None
    assert [p.id for p in s.pairs] == [1, 2]
    assert [p.enabled for p in s.pairs] == [True, False]
    assert [p.colour for p in s.pairs] == ["#fff400", "#b219ff"]
    assert (s.next_id, s.colour_seed, s.status, s.revision) == (13, 4211, "finished", 9)
    assert len(s.load_warnings) == 1 and "fingerprint" in s.load_warnings[0]
    assert s.loaded_content_sha == EXAMPLE_SHA

    p = labelling_to_attrs(s, d, status="draft", software_version="0.4.0")
    assert {k: v for k, v in p["coarse"].items() if k != "matrix_xy"} == {
        k: v for k, v in EXAMPLE_LABELLING["coarse"].items() if k != "matrix_xy"
    }
    assert p["coarse"]["matrix_xy"][0][2] == pytest.approx(78338.8334189, abs=1e-6)
    assert p["crops"] == EXAMPLE_LABELLING["crops"]
    assert p["landmarks"] == EXAMPLE_LABELLING["landmarks"]
    assert p["content_sha256"] == EXAMPLE_SHA
    assert p["revision"] == 9 and p["finished_at"] is None
    assert p["store_grid_digest"] == d.grid_digest
    assert p["images"]["reference"] == {"shape_yx": [55012, 76544], "pixel_size_um": 0.34511}
    assert s.content_sha256() == p["content_sha256"]
    added = s.add_pair((10, 20), (30, 40))
    assert s.content_sha256() != EXAMPLE_SHA
    s.remove_pairs([added.id])
    assert s.content_sha256() == EXAMPLE_SHA
    assert p["status"] == "draft"
    assert labelling_to_attrs(s, d, status="finished", software_version="0.4.0")["finished_at"]


def _descriptor_with_payload(payload=EXAMPLE_LABELLING):
    return parse_store_descriptor(
        Path("PAIR1_prepared.zarr"),
        {STORE_ATTR: prepared_store_attr(), LABELLING_ATTR: payload},
    )


def test_labelling_mismatch_and_errors():
    with pytest.raises(LabellingMismatch, match="start fresh"):
        labelling_from_attrs(
            EXAMPLE_LABELLING,
            parse_store_descriptor(
                Path("x.zarr"), {STORE_ATTR: prepared_store_attr(ref_shape=(55012, 76545))}
            ),
        )
    with pytest.raises(LabellingMismatch):
        labelling_from_attrs(
            EXAMPLE_LABELLING,
            parse_store_descriptor(Path("x.zarr"), {STORE_ATTR: prepared_store_attr(p_m=0.345)}),
        )
    d = _descriptor_with_payload()
    no_images = {k: v for k, v in EXAMPLE_LABELLING.items() if k != "images"}
    with pytest.raises(LabellingMismatch):
        labelling_from_attrs(no_images, d)
    newer = {**EXAMPLE_LABELLING, "schema_version": 2}
    with pytest.raises(UnsupportedSchemaVersion, match="newer"):
        labelling_from_attrs(newer, d)
    with pytest.raises(ValueError, match="not a landmark labelling"):
        labelling_from_attrs({"schema": "x"}, d)
    with pytest.raises(ValueError, match="not a landmark labelling"):
        labelling_from_attrs([], d)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        labelling_from_attrs({**EXAMPLE_LABELLING, "schema_version": "1"}, d)
    dup = copy.deepcopy(EXAMPLE_LABELLING)
    dup["landmarks"]["pairs"][1]["id"] = 1
    with pytest.raises(ValueError, match="duplicate landmark id 1"):
        labelling_from_attrs(dup, d)
    for key, value in [("status", "weird"), ("revision", 0), ("revision", True)]:
        with pytest.raises(ValueError):
            labelling_from_attrs({**EXAMPLE_LABELLING, key: value}, d)
    bad_crop = copy.deepcopy(EXAMPLE_LABELLING)
    bad_crop["crops"]["moving"] = {"x0": 0, "y0": 0, "x1": 999999, "y1": 5}
    with pytest.raises(ValueError, match="invalid moving crop"):
        labelling_from_attrs(bad_crop, d)
    bad_mirror = copy.deepcopy(EXAMPLE_LABELLING)
    bad_mirror["coarse"]["mirror_x"] = 1
    with pytest.raises(ValueError):
        labelling_from_attrs(bad_mirror, d)


def test_labelling_repairs_and_warnings():
    d = _descriptor_with_payload()
    low = copy.deepcopy(EXAMPLE_LABELLING)
    low["landmarks"]["next_id"] = 2
    s = labelling_from_attrs(low, d)
    assert s.next_id == 3
    assert any("next_id" in w for w in s.load_warnings)

    colour = copy.deepcopy(EXAMPLE_LABELLING)
    colour["landmarks"]["pairs"][0]["colour"] = "red"
    s = labelling_from_attrs(colour, d)
    assert s.pairs[0].colour == pair_colour(1, 4211)
    assert any("colour" in w for w in s.load_warnings)

    tampered = {**EXAMPLE_LABELLING, "content_sha256": "sha256:" + "0" * 64}
    s = labelling_from_attrs(tampered, d)
    assert any("content_sha256" in w for w in s.load_warnings)

    outside = copy.deepcopy(EXAMPLE_LABELLING)
    outside["landmarks"]["pairs"][1]["moving_xy_px"] = [1e9, 5.0]
    s = labelling_from_attrs(outside, d)
    assert any("landmark 2" in w for w in s.load_warnings)
    assert s.pairs[1].moving_xy == (1e9, 5.0)

    missing_enabled = copy.deepcopy(EXAMPLE_LABELLING)
    del missing_enabled["landmarks"]["pairs"][1]["enabled"]
    assert labelling_from_attrs(missing_enabled, d).pairs[1].enabled is True

    floor = {**EXAMPLE_LABELLING, "outlier_floor_um": -3}
    s = labelling_from_attrs(floor, d)
    assert s.outlier_floor_um is None
    assert any("outlier_floor_um" in w for w in s.load_warnings)


def test_rounding_of_written_values():
    d = prepared_descriptor()
    s = LabellingState.new(colour_seed=1)
    s.coarse = CoarseState(-179.99996, False, (1.23456789, -0.00001))
    s.add_pair((1.00049, 2.0), (3.0, 4.0))
    p = labelling_to_attrs(s, d, status="draft", software_version="0.4.0")
    assert p["coarse"]["offset_um"] == [1.2346, 0.0]
    assert math.copysign(1.0, p["coarse"]["offset_um"][1]) == 1.0
    assert p["coarse"]["rotation_deg"] == 180.0
    assert p["landmarks"]["pairs"][0]["reference_xy_px"] == [1.0, 2.0]
    assert all(isinstance(v, float) for v in p["landmarks"]["pairs"][0]["moving_xy_px"])
    assert set(p["landmarks"]["pairs"][0]) == {
        "id",
        "colour",
        "enabled",
        "reference_xy_px",
        "moving_xy_px",
    }
    s.coarse = CoarseState(0, False, (float("nan"), 0.0))
    with pytest.raises(ValueError):
        labelling_to_attrs(s, d, status="draft", software_version="0.4.0")
    with pytest.raises(ValueError):
        labelling_to_attrs(s, d, status="bogus", software_version="0.4.0")


def test_labelling_to_attrs_needs_prepared_store():
    res = parse_store_descriptor(
        Path("PAIR1_registered.zarr"),
        {STORE_ATTR: result_store_attr(), M.RESULT_ATTR: RESULT_ATTR_MIN},
    )
    with pytest.raises(ValueError, match="prepared store"):
        labelling_to_attrs(LabellingState.new(1), res, status="draft", software_version="x")


# ---------------------------------------------------------------------------------------------
# Store descriptor
# ---------------------------------------------------------------------------------------------


def test_parse_prepared_descriptor():
    d = parse_store_descriptor(
        Path("PAIR1_prepared.zarr"),
        {STORE_ATTR: prepared_store_attr(p_m=0.5), LABELLING_ATTR: EXAMPLE_LABELLING},
    )
    assert d.kind == "prepared" and d.dataset_name == "PAIR1"
    assert set(d.images) == {"reference", "moving"}
    assert d.images["reference"].element_key == "reference"
    assert d.images["moving"].element_key == "moving"
    assert np.allclose(d.images["moving"].base_affine_rc, np.diag([0.5, 0.5, 1.0]))
    assert np.allclose(d.images["reference"].base_affine_rc, np.diag([0.34511, 0.34511, 1.0]))
    assert d.labelling == EXAMPLE_LABELLING
    assert d.result is None
    assert d.grid_digest == "sha256:" + "a" * 64
    assert d.images["moving"].channel_names == ("DAPI", "AF568", "AF647")


def test_parse_result_descriptor():
    d = parse_store_descriptor(
        Path("PAIR1_registered.zarr"),
        {STORE_ATTR: result_store_attr(), M.RESULT_ATTR: RESULT_ATTR_MIN},
    )
    p = 0.34511
    expected = np.array([[p, 0, p * 512], [0, p, p * 1024], [0, 0, 1]])
    for role in ("reference", "moving"):
        assert np.allclose(d.images[role].base_affine_rc, expected)
    assert d.images["moving"].element_key == "moving_registered"
    assert d.result == RESULT_ATTR_MIN
    assert d.labelling is None
    assert d.grid_digest == "sha256:" + "a" * 64
    # no crop on a result store: origin 0 and a warning
    attr = result_store_attr()
    del attr["images"]["reference"]["crop"]
    d = parse_store_descriptor(Path("r.zarr"), {STORE_ATTR: attr, M.RESULT_ATTR: RESULT_ATTR_MIN})
    assert d.warnings and np.allclose(d.images["reference"].base_affine_rc[:2, 2], 0)


def test_parse_descriptor_errors():
    with pytest.raises(NotARegistrationStore, match="PAIR1_prepared.zarr"):
        parse_store_descriptor(Path("PAIR1_prepared.zarr"), {})
    bad_schema = {**prepared_store_attr(), "schema": "other"}
    with pytest.raises(NotARegistrationStore):
        parse_store_descriptor(Path("x.zarr"), {STORE_ATTR: bad_schema})
    with pytest.raises(NotARegistrationStore):
        parse_store_descriptor(
            Path("x.zarr"), {STORE_ATTR: {**prepared_store_attr(), "schema_version": "1"}}
        )
    with pytest.raises(UnsupportedSchemaVersion, match="newer"):
        parse_store_descriptor(
            Path("x.zarr"), {STORE_ATTR: {**prepared_store_attr(), "schema_version": 2}}
        )
    with pytest.raises(NotARegistrationStore):
        parse_store_descriptor(
            Path("x.zarr"), {STORE_ATTR: {**prepared_store_attr(), "kind": "other"}}
        )
    bad_chan = prepared_store_attr()
    bad_chan["images"]["moving"]["registration_channel"] = "GFP"
    with pytest.raises(NotARegistrationStore, match="images.moving.registration_channel"):
        parse_store_descriptor(Path("x.zarr"), {STORE_ATTR: bad_chan})
    for field, value in [
        ("shape_yx", [0, 5]),
        ("pixel_size_um", 0),
        ("channel_names", []),
        ("element", 3),
    ]:
        attr = prepared_store_attr()
        attr["images"]["reference"][field] = value
        with pytest.raises(NotARegistrationStore, match=f"images.reference.{field}"):
            parse_store_descriptor(Path("x.zarr"), {STORE_ATTR: attr})
    with pytest.raises(NotARegistrationStore, match="did not finish"):
        parse_store_descriptor(Path("r.zarr"), {STORE_ATTR: result_store_attr()})
    with pytest.raises(UnsupportedSchemaVersion):
        parse_store_descriptor(
            Path("r.zarr"),
            {
                STORE_ATTR: result_store_attr(),
                M.RESULT_ATTR: {**RESULT_ATTR_MIN, "schema_version": 2},
            },
        )
    with pytest.raises(UnsupportedSchemaVersion):
        parse_store_descriptor(
            Path("x.zarr"),
            {
                STORE_ATTR: prepared_store_attr(),
                LABELLING_ATTR: {**EXAMPLE_LABELLING, "schema_version": 3},
            },
        )


def test_malformed_labelling_gives_warning():
    d = parse_store_descriptor(
        Path("x.zarr"), {STORE_ATTR: prepared_store_attr(), LABELLING_ATTR: ["junk"]}
    )
    assert d.labelling is None
    assert any("malformed" in w for w in d.warnings)


def test_finish_allowed():
    ok, why = finish_allowed(7, False, False)
    assert not ok and "8" in why and "7" in why
    ok, why = finish_allowed(8, True, False)
    assert not ok and "pending" in why
    ok, why = finish_allowed(8, False, True)
    assert not ok and "Saving" in why
    assert finish_allowed(8, False, False) == (True, "")


# ---------------------------------------------------------------------------------------------
# evaluate_landmarks and state helpers
# ---------------------------------------------------------------------------------------------


def _a2_state(mirror=False):
    true = _true_matrix()
    state = LabellingState.new(colour_seed=7)
    state.coarse = CoarseState(179, mirror, (150, -80))
    for ref, mov in zip(A2_REFERENCE, A2_MOVING, strict=True):
        state.add_pair(ref, mov)
    state.add_pair(tuple(apply_matrix_xy(true, np.array([30000.0, 30000.0]))), (30000, 30000))
    state.set_enabled(11, False)
    return state, true


def _evaluate(state, **kw):
    return evaluate_landmarks(state, moving_pixel_size_um=P, reference_pixel_size_um=P, **kw)


def test_evaluate_with_too_few_pairs():
    state = LabellingState.new(colour_seed=1)
    ev = _evaluate(state)
    assert ev.rows == () and ev.summary is None and ev.matrix is None
    state.add_pair((10, 10), (20, 20))
    ev = _evaluate(state)
    assert ev.summary is None and ev.matrix is None
    assert np.isnan(ev.rows[0].fit_um) and np.isnan(ev.rows[0].loo_um)
    assert not ev.rows[0].used and not ev.rows[0].outlier


def test_evaluate_two_pairs_in_sample():
    state = LabellingState.new(colour_seed=1)
    state.add_pair((100, 100), (10, 10))
    state.add_pair((200, 150), (110, 60))
    ev = _evaluate(state)
    assert ev.summary is not None and ev.matrix is not None
    assert all(r.fit_um == pytest.approx(0.0, abs=1e-6) for r in ev.rows)
    assert all(np.isnan(r.loo_um) for r in ev.rows)
    assert ev.summary.warnings[0].startswith("Leave-one-out residuals need at least 3")


def test_evaluate_a2_state():
    state, true = _a2_state()
    ev = _evaluate(state)
    assert [r.id for r in ev.rows] == list(range(1, 12))
    assert ev.rows[4].outlier and not ev.rows[4].used
    assert not any(r.outlier for i, r in enumerate(ev.rows) if i != 4)
    row11 = ev.rows[10]
    assert not row11.used and not row11.outlier and not row11.enabled
    expected = (
        np.linalg.norm(
            apply_matrix_xy(ev.matrix, np.array([30000.0, 30000.0]))
            - np.array(state.pair(11).reference_xy)
        )
        * P
    )
    assert expected == pytest.approx(0.0915, abs=2e-3)  # close to the true matrix
    assert row11.fit_um == pytest.approx(expected, abs=1e-6)
    assert row11.loo_um == row11.fit_um
    s = ev.summary
    assert (s.n_enabled, s.n_used) == (10, 9)
    assert any("flagged as outliers" in w and "5" in w for w in s.warnings)
    assert s.rms_um == pytest.approx(
        math.sqrt(np.mean(np.array([r.fit_um for r in ev.rows if r.used]) ** 2))
    )
    assert s.expected_scale == pytest.approx(1.0)
    assert not s.reflected


def test_evaluate_outside_crop_and_mirror_warnings():
    state, _ = _a2_state()
    # only pair 6 has a reference x below 11100
    state.crops["reference"] = CropRect(11100, 0, 80000, 61000)
    ev = _evaluate(state)
    assert [r.id for r in ev.rows if r.outside_crop] == [6]
    assert ev.summary.warnings[-1].startswith("1 pair(s) outside the crops: 6.")

    state, _ = _a2_state(mirror=True)
    ev = _evaluate(state)
    assert any("opposite mirror state" in w for w in ev.summary.warnings)


def test_evaluate_scale_and_collinear_warnings():
    state = LabellingState.new(colour_seed=1)
    for i in range(5):
        state.add_pair((i * 1500.0, i * 1500.0), (i * 1000.0, i * 1000.0))
    ev = _evaluate(state)
    w = ev.summary.warnings
    assert any("Fitted scale differs" in x and "+50.0 %" in x for x in w)
    assert any("nearly collinear" in x for x in w)


def test_evaluate_degenerate_geometry():
    state = LabellingState.new(colour_seed=1)
    state.add_pair((1, 1), (5, 5))
    state.add_pair((2, 2), (5, 5))
    ev = _evaluate(state)
    assert ev.summary is None and ev.error and "degenerate" in ev.error
    assert np.isnan(ev.rows[0].fit_um)


def test_outlier_floor_in_attrs_is_outside_digest():
    d = prepared_descriptor()
    state, _ = _a2_state()
    a = labelling_to_attrs(state, d, status="draft", software_version="0.4.0")
    b = labelling_to_attrs(
        state, d, status="draft", software_version="0.4.0", outlier_floor_um=30.0
    )
    assert "outlier_floor_um" not in a
    assert b["outlier_floor_um"] == 30.0
    assert a["content_sha256"] == b["content_sha256"]
    state.revision = 3
    back = labelling_from_attrs({**b, "revision": 3}, d)
    assert back.outlier_floor_um == 30.0


def test_state_pair_helpers():
    s = LabellingState.new(colour_seed=4211)
    p = s.add_pair((1.00049, 2.0), (3.0, 4.0))
    assert p.id == 1 and p.colour == pair_colour(1, 4211) and s.next_id == 2
    assert p.reference_xy == (1.0, 2.0)
    s.add_pair((5, 6), (7, 8))
    removed = s.remove_pairs([1])
    assert [r.id for r in removed] == [1] and [q.id for q in s.pairs] == [2]
    assert s.next_id == 3
    s.restore_pairs(removed)
    assert [q.id for q in s.pairs] == [1, 2]
    assert s.pair(1).colour == pair_colour(1, 4211)
    assert s.next_id == 3
    with pytest.raises(ValueError):
        s.restore_pairs(removed)
    with pytest.raises(KeyError):
        s.pair(99)
    with pytest.raises(KeyError):
        s.set_enabled(99, False)
    s.set_enabled(2, False)
    assert [q.id for q in s.enabled_pairs()] == [1]
    assert LabellingState.new().colour_seed >= 1
    assert LabellingState.new().status == "new"


# ---------------------------------------------------------------------------------------------
# Cross-repo golden vectors (generated by MerXen; never edit or regenerate the file here)
# ---------------------------------------------------------------------------------------------

GOLDEN_PATH = Path(__file__).parent / "data" / "manual_registration_golden_v1.json"
GOLDEN_SHA256 = "ed1e81a1b02fda322b458a3d3dd980ce1d5d3e99bdc4ee6e501bf8ec0ea34164"
RTOL = 1e-9
ATOL = 1e-9


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _array(values):
    """Nested lists where ``null`` means NaN."""
    return np.array(values, dtype=float)


def _coarse(values):
    return CoarseState(
        rotation_deg=values["rotation_deg"],
        mirror_x=values["mirror_x"],
        offset_um=tuple(values["offset_um"]),
    )


def _crop(values):
    return None if values is None else CropRect(**values)


def test_golden_file_digest_is_pinned():
    assert hashlib.sha256(GOLDEN_PATH.read_bytes()).hexdigest() == GOLDEN_SHA256


def test_golden_constants_match_the_module(golden):
    assert len(golden["constants"]) == 12
    for name, value in golden["constants"].items():
        assert getattr(M, name) == value, name


def test_golden_wrap_deg(golden):
    assert len(golden["wrap_deg"]) == 9
    for entry in golden["wrap_deg"]:
        assert abs(wrap_deg(entry["input"]) - entry["output"]) < 1e-9


def test_golden_coarse_cases(golden):
    assert len(golden["coarse_cases"]) == 8
    for case in golden["coarse_cases"]:
        kwargs = {
            "moving_shape_yx": tuple(case["moving_shape_yx"]),
            "reference_shape_yx": tuple(case["reference_shape_yx"]),
            "moving_pixel_size_um": case["moving_pixel_size_um"],
            "reference_pixel_size_um": case["reference_pixel_size_um"],
        }
        matrix = coarse_matrix_xy(_coarse(case["coarse"]), **kwargs)
        np.testing.assert_allclose(matrix, case["matrix_xy"], rtol=RTOL, atol=ATOL)
        recovered, ratio = coarse_from_matrix(np.array(case["matrix_xy"]), **kwargs)
        expected = case["from_matrix"]["coarse"]
        assert abs(wrap_deg(recovered.rotation_deg - expected["rotation_deg"])) < 1e-9
        assert recovered.mirror_x == expected["mirror_x"]
        np.testing.assert_allclose(recovered.offset_um, expected["offset_um"], atol=1e-6)
        assert ratio == pytest.approx(case["from_matrix"]["scale_ratio"], abs=1e-12)


def test_golden_coarse_transitions(golden):
    assert len(golden["coarse_transitions"]) == 70
    for entry in golden["coarse_transitions"]:
        start, arg, action = _coarse(entry["start"]), entry["argument"], entry["action"]
        if action == "flip_horizontal":
            result = start.flipped_horizontal()
        elif action == "flip_vertical":
            result = start.flipped_vertical()
        elif action == "rotate":
            result = start.rotated(arg)
        elif action == "set_rotation":
            result = start.with_rotation(arg)
        elif action == "shift":
            result = start.shifted(arg[0], arg[1])
        else:
            assert action == "reset"
            result = CoarseState()
        expected = entry["result"]
        assert abs(wrap_deg(result.rotation_deg - expected["rotation_deg"])) < 1e-9, entry
        assert result.mirror_x == expected["mirror_x"], entry
        np.testing.assert_allclose(result.offset_um, expected["offset_um"], atol=1e-9)


def test_golden_snap_crop(golden):
    assert len(golden["snap_crop"]) == 8
    for entry in golden["snap_crop"]:
        crop, shape = _crop(entry["crop"]), tuple(entry["shape_yx"])
        kwargs = {"multiple": entry["multiple"], "min_size_px": entry["min_size_px"]}
        if entry["error"]:
            with pytest.raises(ValueError):
                snap_crop(crop, shape, **kwargs)
        else:
            assert snap_crop(crop, shape, **kwargs) == _crop(entry["effective"]), entry["name"]
    # Skipped on purpose (the viewer never needs them): crop_frames (crop/full matrix
    # conversion), flows and file fingerprints.


def test_golden_points_outside_crop(golden):
    for entry in golden["points_outside_crop"]:
        outside = points_outside_crop(np.array(entry["points_xy"]), _crop(entry["crop"]))
        assert outside.tolist() == entry["outside"]


def test_golden_napari_affine(golden):
    for entry in golden["napari_affine"]:
        affine = napari_affine_rc(
            np.array(entry["reference_base_rc"]), np.array(entry["matrix_xy"])
        )
        np.testing.assert_allclose(affine, entry["affine_rc"], rtol=RTOL, atol=ATOL)


def test_golden_fits(golden):
    assert len(golden["fits"]) == 5
    for entry in golden["fits"]:
        moving, reference = _array(entry["moving_xy"]), _array(entry["reference_xy"])
        fit = fit_similarity(moving, reference, reflection=entry["reflection"])
        np.testing.assert_allclose(fit.matrix, entry["matrix"], rtol=RTOL, atol=ATOL)
        assert fit.scale == pytest.approx(entry["scale"], rel=RTOL)
        assert abs(wrap_deg(fit.rotation_deg - entry["rotation_deg"])) < 1e-9
        assert fit.reflected == entry["reflected"]
        np.testing.assert_allclose(fit.residuals_px, entry["residuals_px"], atol=1e-6)
        loo = leave_one_out_residuals(moving, reference, reflection=entry["reflection"])
        expected = _array(entry["loo_px"])
        assert np.isnan(loo).tolist() == np.isnan(expected).tolist()
        np.testing.assert_allclose(loo, expected, atol=1e-6, equal_nan=True)


def test_golden_spread_ratio(golden):
    for entry in golden["spread_ratio"]:
        assert spread_ratio(entry["points_xy"]) == pytest.approx(entry["value"], abs=1e-9)


def test_golden_robust_fits(golden):
    assert len(golden["robust_fits"]) == 7
    for entry in golden["robust_fits"]:
        name = entry["name"]
        moving, reference = _array(entry["moving_xy"]), _array(entry["reference_xy"])
        result = robust_similarity(
            moving,
            reference,
            reflection=entry["reflection"],
            moving_pixel_size_um=entry["moving_pixel_size_um"],
            reference_pixel_size_um=entry["reference_pixel_size_um"],
            options=RobustFitOptions(**entry["options"]),
        )
        np.testing.assert_allclose(
            result.fit.matrix, entry["matrix"], rtol=RTOL, atol=ATOL, err_msg=name
        )
        assert result.fit.scale == pytest.approx(entry["scale"], rel=RTOL), name
        assert abs(wrap_deg(result.fit.rotation_deg - entry["rotation_deg"])) < 1e-9, name
        assert result.used.tolist() == entry["used"], name
        assert result.outlier.tolist() == entry["outlier"], name
        for key, actual in (("residuals_um", result.residuals_um), ("loo_um", result.loo_um)):
            expected = _array(entry[key])
            assert np.isnan(actual).tolist() == np.isnan(expected).tolist(), (name, key)
            np.testing.assert_allclose(actual, expected, atol=1e-6, equal_nan=True)
        first = (
            leave_one_out_residuals(moving, reference, reflection=entry["reflection"])
            * entry["reference_pixel_size_um"]
        )
        expected_first = _array(entry["first_pass_loo_um"])
        assert np.isnan(first).tolist() == np.isnan(expected_first).tolist(), name
        np.testing.assert_allclose(first, expected_first, atol=1e-6, equal_nan=True)
        for key, value in (
            ("chosen_class_median_um", result.chosen_class_median_um),
            ("other_class_median_um", result.other_class_median_um),
        ):
            assert value == pytest.approx(entry[key], abs=1e-6), (name, key)
        assert result.spread_ratio == pytest.approx(entry["spread_ratio"], abs=1e-9)
        assert result.scale_ratio == pytest.approx(entry["scale_ratio"], rel=RTOL)


def test_golden_nan_loo_case_is_present(golden):
    entries = {entry["name"]: entry for entry in golden["robust_fits"]}
    trimmed = entries["nan_loo_trimmed"]
    assert trimmed["loo_um"][6] is None and trimmed["first_pass_loo_um"][6] is None
    assert trimmed["outlier"] == [False, False, False, False, False, True, False]
    assert entries["nan_loo_no_trimming"]["outlier"] == trimmed["outlier"]


def test_golden_pair_colours(golden):
    assert len(golden["pair_colour"]) == 24
    for entry in golden["pair_colour"]:
        assert pair_colour(entry["pair_id"], entry["seed"]) == entry["colour"]


def test_golden_canonical(golden):
    assert len(golden["canonical"]) == 2
    for entry in golden["canonical"]:
        assert canonical_json(entry["object"]) == entry["canonical_json"]
        assert canonical_sha256(entry["object"]) == entry["sha256"]


def test_golden_examples_parse_and_digest(golden):
    examples = golden["examples"]
    labelling = examples["labelling"]
    digest = labelling_content_sha256(labelling)
    assert (
        digest == labelling["content_sha256"] == examples["labelling_content_sha256"] == EXAMPLE_SHA
    )
    prepared = parse_store_descriptor(
        Path("golden_prepared.zarr"),
        {STORE_ATTR: examples["store_attr_prepared"], LABELLING_ATTR: labelling},
    )
    result = parse_store_descriptor(
        Path("golden_registered.zarr"),
        {STORE_ATTR: examples["store_attr_result"], M.RESULT_ATTR: RESULT_ATTR_MIN},
    )
    assert (prepared.kind, result.kind) == ("prepared", "result")
    assert prepared.grid_digest == labelling["store_grid_digest"]
    state = labelling_from_attrs(labelling, prepared)
    assert state.load_warnings == ()
    assert state.outlier_floor_um == 30.0
    rebuilt = labelling_to_attrs(
        state, prepared, status="finished", software_version="0.4.0", outlier_floor_um=30.0
    )
    assert rebuilt["content_sha256"] == labelling["content_sha256"]
    for key in ("rotation_deg", "mirror_x", "offset_um"):
        assert rebuilt["coarse"][key] == labelling["coarse"][key]
    np.testing.assert_allclose(
        rebuilt["coarse"]["matrix_xy"], labelling["coarse"]["matrix_xy"], atol=1e-6
    )
