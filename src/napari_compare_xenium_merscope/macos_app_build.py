"""Private py2app build definition used by the macOS CLI launcher."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

from setuptools import setup


PACKAGE_DIR = Path(__file__).resolve().parent
APP_NAME = "Napari Compare Xenium MERSCOPE"


def main() -> None:
    setup(
        name=APP_NAME,
        version=version("napari-compare-xenium-merscope"),
        app=[
            {
                "script": str(PACKAGE_DIR / "macos_app_entry.py"),
                "dest_base": APP_NAME,
            }
        ],
        options={
            "py2app": {
                "argv_emulation": False,
                "iconfile": str(PACKAGE_DIR / "assets" / "app_icon.icns"),
                "no_chdir": True,
                "plist": {
                    "CFBundleDisplayName": APP_NAME,
                    "CFBundleIdentifier": "org.napari.compare-xenium-merscope",
                    "CFBundleName": APP_NAME,
                    "CFBundleShortVersionString": version("napari-compare-xenium-merscope"),
                    "LSMinimumSystemVersion": "13.0",
                    "NSHighResolutionCapable": True,
                },
            }
        },
    )


if __name__ == "__main__":
    main()
