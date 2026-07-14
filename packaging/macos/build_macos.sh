#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "The macOS package must be built on macOS." >&2
    exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
spec_file="$project_root/packaging/pyinstaller/napari_compare.spec"
dist_dir="$project_root/dist"
work_dir="$project_root/build/pyinstaller-macos"
artifacts_dir="$project_root/artifacts"
app_name="Napari Compare Xenium MERSCOPE.app"
app_path="$dist_dir/$app_name"
executable_path="$app_path/Contents/MacOS/NapariCompareXeniumMERSCOPE"
version="$(python -c "from importlib.metadata import version; print(version('napari-compare-xenium-merscope'))")"
architecture="$(uname -m)"

case "$architecture" in
    arm64|x86_64) ;;
    *)
        echo "Unsupported macOS architecture: $architecture" >&2
        exit 1
        ;;
esac

rm -rf "$work_dir" "$dist_dir/NapariCompareXeniumMERSCOPE" "$app_path"
mkdir -p "$artifacts_dir"
python -m PyInstaller \
    --noconfirm \
    --clean \
    --distpath "$dist_dir" \
    --workpath "$work_dir" \
    "$spec_file"

if [[ ! -x "$executable_path" ]]; then
    echo "PyInstaller did not produce $executable_path" >&2
    exit 1
fi

plist="$app_path/Contents/Info.plist"
bundle_version="$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$plist")"
if [[ "$bundle_version" != "$version" ]]; then
    echo "Application bundle version $bundle_version does not match package version $version" >&2
    exit 1
fi

# PyInstaller ad-hoc signs native code by default. A Developer ID identity can
# be supplied by a release environment to replace that signature.
if [[ -n "${MACOS_SIGNING_IDENTITY:-}" ]]; then
    codesign \
        --force \
        --deep \
        --options runtime \
        --timestamp \
        --sign "$MACOS_SIGNING_IDENTITY" \
        "$app_path"
fi
codesign --verify --deep --strict "$app_path"

dmg_staging="$project_root/build/macos-dmg-root"
dmg_name="Napari-Compare-Xenium-MERSCOPE-${version}-macOS-${architecture}.dmg"
dmg_path="$artifacts_dir/$dmg_name"
rm -rf "$dmg_staging"
mkdir -p "$dmg_staging"
ditto "$app_path" "$dmg_staging/$app_name"
ln -s /Applications "$dmg_staging/Applications"
rm -f "$dmg_path"
hdiutil create \
    -volname "Napari Compare Xenium MERSCOPE" \
    -srcfolder "$dmg_staging" \
    -format UDZO \
    -ov \
    "$dmg_path"

if [[ -n "${MACOS_SIGNING_IDENTITY:-}" ]]; then
    codesign --force --timestamp --sign "$MACOS_SIGNING_IDENTITY" "$dmg_path"
fi

if [[ -n "${MACOS_NOTARY_PROFILE:-}" ]]; then
    if [[ -z "${MACOS_SIGNING_IDENTITY:-}" ]]; then
        echo "MACOS_NOTARY_PROFILE requires MACOS_SIGNING_IDENTITY." >&2
        exit 1
    fi
    xcrun notarytool submit "$dmg_path" \
        --keychain-profile "$MACOS_NOTARY_PROFILE" \
        --wait
    xcrun stapler staple "$dmg_path"
    xcrun stapler validate "$dmg_path"
fi

echo "macOS application: $app_path"
echo "macOS disk image: $dmg_path"
