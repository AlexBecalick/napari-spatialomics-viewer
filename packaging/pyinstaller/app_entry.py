"""Entry point for self-contained Windows and Linux application bundles."""

from __future__ import annotations

import multiprocessing


def main() -> None:
    multiprocessing.freeze_support()
    from napari_compare_xenium_merscope.viewer import main as viewer_main

    viewer_main()


if __name__ == "__main__":
    main()
