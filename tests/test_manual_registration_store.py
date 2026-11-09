"""Tests for the manual-registration store I/O (no Qt)."""

from __future__ import annotations

import json
import logging
import re
import time
import warnings

import pytest
import zarr

from napari_compare_xenium_merscope import manual_registration_store as store_module
from napari_compare_xenium_merscope.manual_registration_model import (
    LABELLING_ATTR,
    LABELLING_SCHEMA,
    STORE_ATTR,
    LabellingState,
    NotARegistrationStore,
    UnsupportedSchemaVersion,
    labelling_content_sha256,
    labelling_to_attrs,
    parse_store_descriptor,
)
from napari_compare_xenium_merscope.manual_registration_store import (
    StoreBusyError,
    StoreChangedError,
    clear_local_draft,
    draft_path,
    history_dir,
    read_local_draft,
    read_root_attrs,
    read_store_descriptor,
    save_labelling,
    store_write_lock,
    write_local_draft,
    write_lock_path,
)

fcntl = store_module.fcntl


def prepared_store_attr(kind="prepared"):
    def rec(role, shape):
        return {
            "element": role,
            "role": role,
            "shape_yx": list(shape),
            "dtype": "uint16",
            "pixel_size_um": 0.34511,
            "channel_names": ["DAPI", "AF568"],
            "registration_channel": "DAPI",
        }

    attr = {
        "schema": "merxen.manual_registration.store",
        "schema_version": 1,
        "kind": kind,
        "dataset_name": "PAIR1",
        "images": {
            "reference": rec("reference", (55012, 76544)),
            "moving": rec("moving", (61003, 80112)),
        },
        "grid_digest": "sha256:" + "a" * 64,
        "content_digest": "sha256:" + "b" * 64,
    }
    if kind == "result":
        attr["prepared_store"] = {"grid_digest": "sha256:" + "a" * 64}
    return attr


def make_store(tmp_path, attrs=None, zarr_format=3, consolidated=True, name="PAIR1_prepared.zarr"):
    p = tmp_path / name
    attrs = {STORE_ATTR: prepared_store_attr()} if attrs is None else attrs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = zarr.open_group(p, mode="w", zarr_format=zarr_format)
        g.attrs.update(attrs)
        g.attrs["merxen_schema"] = {"v": 1}
        g.attrs["spatialdata_attrs"] = {"version": "0.2"}
        g.create_array("images/reference", shape=(1, 64, 64), dtype="uint16")
        if consolidated:
            zarr.consolidate_metadata(p)
    return p


def descriptor_for(path):
    return read_store_descriptor(path)


def make_payload(path, n_pairs=8, status="finished", seed=4211, revision=None):
    d = descriptor_for(path)
    state = LabellingState.new(colour_seed=seed)
    for i in range(n_pairs):
        state.add_pair((100.0 + 50 * i, 200.0 + 30 * i), (400.0 + 20 * i, 90.0 + 70 * i))
    state.revision = revision
    return labelling_to_attrs(state, d, status=status, software_version="0.4.0")


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------


def test_read_root_attrs_v3_consolidated(tmp_path):
    p = make_store(tmp_path)
    attrs = read_root_attrs(p)
    assert STORE_ATTR in attrs and attrs["merxen_schema"] == {"v": 1}
    assert attrs["spatialdata_attrs"] == {"version": "0.2"}
    assert read_store_descriptor(p).kind == "prepared"


def test_read_root_attrs_v2(tmp_path):
    p = make_store(tmp_path, zarr_format=2)
    assert (p / ".zattrs").exists()
    attrs = read_root_attrs(p)
    assert STORE_ATTR in attrs and attrs["merxen_schema"] == {"v": 1}


def test_read_root_attrs_errors(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(NotARegistrationStore):
        read_root_attrs(plain)
    with pytest.raises(FileNotFoundError):
        read_root_attrs(tmp_path / "missing.zarr")
    bad = tmp_path / "bad.zarr"
    bad.mkdir()
    (bad / "zarr.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable zarr.json"):
        read_root_attrs(bad)
    v2 = tmp_path / "v2.zarr"
    v2.mkdir()
    (v2 / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")
    assert read_root_attrs(v2) == {}


# ---------------------------------------------------------------------------------------------
# Save protocol
# ---------------------------------------------------------------------------------------------


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_save_protocol_v3_consolidated(tmp_path):
    p = make_store(tmp_path)
    before = _json(p / "zarr.json")
    payload = make_payload(p)
    out = save_labelling(p, payload, expected_revision=None, software_version="0.4.0")
    assert out["revision"] == 1
    assert out["finished_at"] == out["updated_at"]
    assert out["written_by"] == {"software": "napari-compare-xenium-merscope", "version": "0.4.0"}
    assert out["content_sha256"] == labelling_content_sha256(out)
    after = _json(p / "zarr.json")
    for key in ("merxen_schema", "spatialdata_attrs", STORE_ATTR):
        assert after["attributes"][key] == before["attributes"][key]
    assert LABELLING_ATTR in after["attributes"]
    assert "images/reference" in after["consolidated_metadata"]["metadata"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert zarr.open_group(p, mode="r").attrs[LABELLING_ATTR] == out
    assert not history_dir(p).exists()
    # the payload argument is left untouched
    assert payload["revision"] == 0


def test_second_save_backs_up_and_keeps_finished_at(tmp_path):
    p = make_store(tmp_path)
    first = save_labelling(p, make_payload(p), expected_revision=None, software_version="0.4.0")
    second = save_labelling(
        p, make_payload(p, status="draft", n_pairs=3), expected_revision=1, software_version="0.4.1"
    )
    assert second["revision"] == 2
    assert second["finished_at"] == first["finished_at"]
    assert second["written_by"]["version"] == "0.4.1"
    files = sorted(history_dir(p).iterdir())
    assert len(files) == 1 and re.fullmatch(r"1-\d{8}T\d{6}Z\.json", files[0].name)
    assert _json(files[0]) == first


def test_stale_and_missing_revision(tmp_path):
    p = make_store(tmp_path)
    save_labelling(p, make_payload(p), expected_revision=None, software_version="0.4.0")
    save_labelling(
        p, make_payload(p, status="draft"), expected_revision=1, software_version="0.4.0"
    )
    with pytest.raises(StoreChangedError, match="another session"):
        save_labelling(
            p, make_payload(p, status="draft"), expected_revision=1, software_version="0.4.0"
        )
    assert read_root_attrs(p)[LABELLING_ATTR]["revision"] == 2
    with pytest.raises(StoreChangedError):
        save_labelling(
            p, make_payload(p, status="draft"), expected_revision=None, software_version="0.4.0"
        )


def test_save_protocol_v2_consolidated(tmp_path):
    p = make_store(tmp_path, zarr_format=2)
    out1 = save_labelling(p, make_payload(p), expected_revision=None, software_version="0.4.0")
    out2 = save_labelling(
        p, make_payload(p, status="draft"), expected_revision=1, software_version="0.4.0"
    )
    assert _json(p / ".zattrs")[LABELLING_ATTR] == out2
    zmeta = _json(p / ".zmetadata")
    assert zmeta["metadata"][".zattrs"][LABELLING_ATTR] == out2
    assert zmeta["metadata"][".zattrs"]["merxen_schema"] == {"v": 1}
    assert out2["revision"] == 2 and out1["revision"] == 1
    assert len(list(history_dir(p).iterdir())) == 1


def test_rejected_payloads_write_nothing(tmp_path):
    p = make_store(tmp_path)
    original = (p / "zarr.json").read_bytes()
    with pytest.raises(ValueError, match="at least 8"):
        save_labelling(p, make_payload(p, n_pairs=7), expected_revision=None, software_version="x")
    nan_payload = make_payload(p, status="draft")
    nan_payload["landmarks"]["pairs"][0]["moving_xy_px"][0] = float("nan")
    with pytest.raises(ValueError):
        save_labelling(p, nan_payload, expected_revision=None, software_version="x")
    wrong = make_payload(p, status="draft")
    wrong["schema"] = "other"
    with pytest.raises(ValueError):
        save_labelling(p, wrong, expected_revision=None, software_version="x")
    assert (p / "zarr.json").read_bytes() == original


def test_result_store_and_rebuilt_store_are_refused(tmp_path):
    prepared = make_store(tmp_path)
    payload = make_payload(prepared, status="draft")
    result = make_store(
        tmp_path,
        {STORE_ATTR: prepared_store_attr("result"), "merxen_manual_registration_result": {}},
        name="PAIR1_registered.zarr",
    )
    with pytest.raises(NotARegistrationStore):
        save_labelling(result, payload, expected_revision=None, software_version="x")
    # grid digest changed on disk
    rebuilt_attr = prepared_store_attr()
    rebuilt_attr["grid_digest"] = "sha256:" + "c" * 64
    rebuilt = make_store(tmp_path, {STORE_ATTR: rebuilt_attr}, name="PAIR2_prepared.zarr")
    with pytest.raises(StoreChangedError, match="rebuilt"):
        save_labelling(rebuilt, payload, expected_revision=None, software_version="x")
    # a plain directory is never turned into a store
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(NotARegistrationStore):
        save_labelling(plain, payload, expected_revision=None, software_version="x")
    assert not list(plain.iterdir())


def test_newer_on_disk_labelling_is_not_overwritten(tmp_path):
    attrs = {
        STORE_ATTR: prepared_store_attr(),
        LABELLING_ATTR: {"schema": LABELLING_SCHEMA, "schema_version": 2, "revision": 4},
    }
    p = make_store(tmp_path, attrs)
    with pytest.raises(UnsupportedSchemaVersion):
        save_labelling(
            p, make_payload(p, status="draft"), expected_revision=4, software_version="x"
        )
    assert read_root_attrs(p)[LABELLING_ATTR]["schema_version"] == 2


def test_backup_failure_is_only_a_warning(tmp_path, caplog):
    p = make_store(tmp_path)
    save_labelling(p, make_payload(p), expected_revision=None, software_version="0.4.0")
    history_dir(p).write_text("in the way", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=store_module.__name__):
        out = save_labelling(
            p, make_payload(p, status="draft"), expected_revision=1, software_version="0.4.0"
        )
    assert out["revision"] == 2
    assert any("could not back up" in r.message for r in caplog.records)


def test_save_without_fcntl_takes_no_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "fcntl", None)
    p = make_store(tmp_path)
    out = save_labelling(p, make_payload(p), expected_revision=None, software_version="0.4.0")
    assert out["revision"] == 1
    assert not write_lock_path(p).exists()


@pytest.mark.skipif(fcntl is None, reason="needs fcntl")
def test_lock_contention(tmp_path):
    p = make_store(tmp_path)
    lock = write_lock_path(p)
    with lock.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        start = time.monotonic()
        with pytest.raises(StoreBusyError, match="busy"):
            with store_write_lock(p, timeout_s=0.2):
                pass
        assert time.monotonic() - start < 1.0
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    with store_write_lock(p, timeout_s=0.2):
        pass
    link = tmp_path / "link.zarr"
    link.symlink_to(p, target_is_directory=True)
    assert write_lock_path(link) == write_lock_path(p)
    assert history_dir(link) == history_dir(p)


@pytest.mark.skipif(fcntl is None, reason="needs fcntl")
def test_lock_is_not_reentrant(tmp_path):
    p = make_store(tmp_path)
    with store_write_lock(p, timeout_s=0.2):
        with pytest.raises(StoreBusyError):
            with store_write_lock(p, timeout_s=0.1):
                pass


def test_read_back_mismatch_raises(tmp_path, monkeypatch):
    p = make_store(tmp_path)
    real = store_module.read_root_attrs

    def tampered(path):
        attrs = real(path)
        if LABELLING_ATTR in attrs:
            attrs[LABELLING_ATTR] = {**attrs[LABELLING_ATTR], "status": "tampered"}
        return attrs

    monkeypatch.setattr(store_module, "read_root_attrs", tampered)
    with pytest.raises(RuntimeError, match="read back identically"):
        save_labelling(p, make_payload(p), expected_revision=None, software_version="0.4.0")


def test_saved_payload_round_trips_through_the_model(tmp_path):
    p = make_store(tmp_path)
    save_labelling(p, make_payload(p), expected_revision=None, software_version="0.4.0")
    d = read_store_descriptor(p)
    assert d.labelling is not None and d.labelling["revision"] == 1
    from napari_compare_xenium_merscope.manual_registration_model import labelling_from_attrs

    state = labelling_from_attrs(d.labelling, d)
    assert state.status == "finished" and state.revision == 1 and len(state.pairs) == 8
    assert state.load_warnings == ()
    assert parse_store_descriptor(p, read_root_attrs(p)).labelling == d.labelling


# ---------------------------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------------------------


def test_drafts(tmp_path):
    base = tmp_path / "cfg"
    store_a = make_store(tmp_path, name="A_prepared.zarr")
    store_b = make_store(tmp_path, name="B_prepared.zarr")
    payload = make_payload(store_a, status="draft", revision=1)
    assert read_local_draft(store_a, base) is None
    write_local_draft(store_a, base, payload)
    draft = read_local_draft(store_a, base)
    assert draft["payload"] == payload
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", draft["saved_at"])
    assert draft_path(store_a, base) != draft_path(store_b, base)
    assert read_local_draft(store_b, base) is None
    link = tmp_path / "link.zarr"
    link.symlink_to(store_a, target_is_directory=True)
    assert draft_path(link, base) == draft_path(store_a, base)
    assert read_local_draft(link, base)["payload"] == payload
    # overwriting leaves no temporary files behind
    write_local_draft(store_a, base, payload)
    assert [f.name for f in draft_path(store_a, base).parent.iterdir()] == [
        draft_path(store_a, base).name
    ]


def test_draft_rejections(tmp_path):
    base = tmp_path / "cfg"
    store = make_store(tmp_path)
    target = draft_path(store, base)
    write_local_draft(store, base, {"a": 1})
    good = json.loads(target.read_text(encoding="utf-8"))
    target.write_text("{corrupt", encoding="utf-8")
    assert read_local_draft(store, base) is None
    target.write_bytes(b"\xff\xfe\x80not utf-8")  # damaged bytes must not raise
    assert read_local_draft(store, base) is None
    for patch in ({"draft_format": 2}, {"payload": [1]}, {"store_path": "elsewhere"}):
        target.write_text(json.dumps({**good, **patch}), encoding="utf-8")
        assert read_local_draft(store, base) is None
    target.write_text(json.dumps(good), encoding="utf-8")
    assert read_local_draft(store, base) is not None
    clear_local_draft(store, base)
    assert not target.exists()
    clear_local_draft(store, base)  # a second call does not raise


def test_draft_write_errors_propagate(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("a file", encoding="utf-8")
    with pytest.raises(OSError):
        write_local_draft(tmp_path / "x.zarr", blocker, {"a": 1})
    with pytest.raises(ValueError):
        write_local_draft(tmp_path / "x.zarr", tmp_path / "cfg", {"a": float("nan")})
    assert not list((tmp_path / "cfg").rglob("*.tmp"))
