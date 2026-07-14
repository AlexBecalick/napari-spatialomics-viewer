"""Cross-platform one-directory PyInstaller definition."""

from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata


SPEC_DIR = Path(SPEC).resolve().parent
PROJECT_ROOT = SPEC_DIR.parent.parent
PACKAGE_ROOT = PROJECT_ROOT / "src"
APP_NAME = "NapariCompareXeniumMERSCOPE"

datas = []
binaries = []
hiddenimports = []


def is_runtime_module(module_name):
    """Exclude package test helpers from the distributable application."""
    parts = set(module_name.split("."))
    return not parts.intersection({"tests", "_tests", "testing", "conftest"})

# napari, SpatialData and the Zarr/xarray stack use dynamic imports and entry
# points extensively. Collecting their package resources here makes the frozen
# application independent of a user's Python installation.
datas += collect_data_files("napari_compare_xenium_merscope")

for package_name in (
    "napari",
    "napari_builtins",
    "napari_console",
    "napari_svg",
    "spatialdata",
    "ome_zarr",
    "vispy",
    "xarray",
    "zarr",
    "numcodecs",
):
    package_datas, package_binaries, package_hiddenimports = collect_all(
        package_name,
        filter_submodules=is_runtime_module,
        exclude_datas=["**/tests/**", "**/_tests/**", "**/testing/**"],
    )
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

for distribution_name in (
    "napari",
    "napari-console",
    "napari-svg",
    "napari-compare-xenium-merscope",
    "spatialdata",
    "ome-zarr",
    "xarray",
    "zarr",
):
    datas += copy_metadata(distribution_name, recursive=True)

analysis = Analysis(
    [str(SPEC_DIR / "app_entry.py")],
    pathex=[str(PACKAGE_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt6", "PySide2", "PySide6", "pytest"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=(
        str(PACKAGE_ROOT / "napari_compare_xenium_merscope" / "assets" / "app_icon.ico")
        if sys.platform == "win32"
        else None
    ),
)

bundle = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)
