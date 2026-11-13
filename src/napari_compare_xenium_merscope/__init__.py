"""Napari viewer for comparing Xenium and MERSCOPE SpatialData outputs and viewing Atera stores."""

from importlib.metadata import PackageNotFoundError, version as _package_version

try:
    # pyproject.toml holds the only copy of the version (docs/versioning.md).
    __version__ = _package_version("napari-compare-xenium-merscope")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"

__all__ = ["__version__"]
