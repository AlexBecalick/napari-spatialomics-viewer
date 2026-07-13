"""Platform launcher that routes macOS through a genuine application bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

from .macos_app_entry import APP_SUPPORT_DIR


APP_NAME = "Napari Compare Xenium MERSCOPE"
BUNDLE_PATH = Path.home() / "Applications" / f"{APP_NAME}.app"
BUNDLE_MANIFEST_PATH = APP_SUPPORT_DIR / "bundle_environment.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_manifest() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    return {
        "python": str(Path(sys.executable).resolve()),
        "package": str(package_dir),
        "version": version("napari-compare-xenium-merscope"),
        "build_sha256": _file_sha256(package_dir / "macos_app_build.py"),
        "entry_sha256": _file_sha256(package_dir / "macos_app_entry.py"),
        "icon_sha256": _file_sha256(package_dir / "assets" / "app_icon.icns"),
    }


def _bundle_executable(bundle: Path) -> Path:
    return bundle / "Contents" / "MacOS" / APP_NAME


def macos_bundle_is_current(bundle: Path = BUNDLE_PATH) -> bool:
    executable = _bundle_executable(bundle)
    if not executable.is_file() or not os.access(executable, os.X_OK) or not BUNDLE_MANIFEST_PATH.is_file():
        return False
    try:
        manifest = json.loads(BUNDLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    return manifest == _expected_manifest()


def build_macos_app_bundle(destination: Path = BUNDLE_PATH) -> Path:
    """Build and install an alias-mode `.app` for the active Python environment."""
    if sys.platform != "darwin":
        raise RuntimeError("The macOS application bundle can only be built on macOS.")

    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="napari-compare-macos-app-") as temporary:
        temporary_path = Path(temporary)
        dist_dir = temporary_path / "dist"
        build_dir = temporary_path / "build"
        command = [
            sys.executable,
            "-m",
            "napari_compare_xenium_merscope.macos_app_build",
            "py2app",
            "--alias",
            "--dist-dir",
            str(dist_dir),
            "--bdist-base",
            str(build_dir),
        ]
        subprocess.run(command, check=True, cwd=temporary_path)
        built_bundle = dist_dir / f"{APP_NAME}.app"
        if not _bundle_executable(built_bundle).is_file():
            raise RuntimeError(f"py2app did not produce the expected bundle: {built_bundle}")

        staged = destination.with_name(f".{destination.name}.new")
        previous = destination.with_name(f".{destination.name}.previous")
        shutil.rmtree(staged, ignore_errors=True)
        shutil.rmtree(previous, ignore_errors=True)
        shutil.copytree(built_bundle, staged, symlinks=True)
        if destination.exists():
            destination.rename(previous)
        staged.rename(destination)
        shutil.rmtree(previous, ignore_errors=True)
        APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
        BUNDLE_MANIFEST_PATH.write_text(
            json.dumps(_expected_manifest(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        launch_services = Path(
            "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
            "LaunchServices.framework/Support/lsregister"
        )
        if launch_services.is_file():
            subprocess.run(
                [str(launch_services), "-f", str(destination)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    return destination


def ensure_macos_app_bundle() -> Path:
    if macos_bundle_is_current():
        return BUNDLE_PATH
    return build_macos_app_bundle()


def install_macos_app_bundle() -> None:
    """Console-script entry point for explicitly rebuilding the application."""
    bundle = build_macos_app_bundle()
    print(f"Installed {bundle}")


def main() -> None:
    if sys.platform != "darwin":
        from .viewer import main as viewer_main

        viewer_main()
        return

    try:
        bundle = ensure_macos_app_bundle()
    except Exception as exc:
        print(
            f"WARNING: Could not create the macOS application bundle ({exc}); "
            "launching directly through Python.",
            file=sys.stderr,
        )
        from .viewer import main as viewer_main

        viewer_main()
        return
    arguments = list(sys.argv[1:])
    executable = _bundle_executable(bundle)
    os.execv(str(executable), [str(executable), *arguments])
