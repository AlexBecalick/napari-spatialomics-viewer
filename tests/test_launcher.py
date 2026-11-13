from __future__ import annotations

import json

from napari_compare_xenium_merscope import launcher


def test_bundle_current_requires_matching_environment_manifest(tmp_path, monkeypatch):
    bundle = tmp_path / "Viewer.app"
    executable = bundle / "Contents" / "MacOS" / launcher.APP_NAME
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"launcher")
    executable.chmod(0o755)
    manifest_path = tmp_path / "bundle_environment.json"
    expected = {"python": "/test/python", "version": "1"}
    monkeypatch.setattr(launcher, "BUNDLE_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(launcher, "_expected_manifest", lambda: expected)

    assert not launcher.macos_bundle_is_current(bundle)
    manifest_path.write_text(json.dumps(expected), encoding="utf-8")
    assert launcher.macos_bundle_is_current(bundle)
    manifest_path.write_text(json.dumps({**expected, "version": "2"}), encoding="utf-8")
    assert not launcher.macos_bundle_is_current(bundle)


def test_version_flag_prints_the_package_version_without_launching(monkeypatch, capsys):
    import napari_compare_xenium_merscope

    monkeypatch.setattr(launcher.sys, "argv", ["napari-compare-xenium-merscope", "--version"])
    monkeypatch.setattr(
        launcher, "ensure_macos_app_bundle", lambda: (_ for _ in ()).throw(AssertionError("bundle built"))
    )
    launcher.main()
    assert capsys.readouterr().out.strip() == (
        f"napari-compare-xenium-merscope {napari_compare_xenium_merscope.__version__}"
    )
