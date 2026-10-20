"""Ignore Finder metadata without hiding unexpected Zarr entries."""
import warnings

import zarr

from napari_compare_xenium_merscope.viewer import _ignore_macos_zarr_metadata_warning


def test_finder_metadata_warning_is_ignored_but_unknown_entries_warn(tmp_path):
    root = zarr.open_group(str(tmp_path / "data.zarr"), mode="w")
    root.create_group("images").create_group("image")
    folder = tmp_path / "data.zarr" / "images"
    (folder / ".DS_Store").write_bytes(b"Finder metadata")
    (folder / "unexpected.txt").write_text("unexpected")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ignore_macos_zarr_metadata_warning()
        assert list(root["images"].group_keys()) == ["image"]
    messages = [str(item.message) for item in caught]
    assert not any(".DS_Store" in message for message in messages)
    assert any("unexpected.txt" in message for message in messages)
    assert (folder / ".DS_Store").exists()
