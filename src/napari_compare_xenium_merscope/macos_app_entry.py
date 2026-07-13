"""Entry point executed by the py2app macOS application bundle."""

from __future__ import annotations

from pathlib import Path


APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "Napari Compare Xenium MERSCOPE"


def main() -> None:
    from napari_compare_xenium_merscope.viewer import main as viewer_main

    viewer_main()


if __name__ == "__main__":
    main()
