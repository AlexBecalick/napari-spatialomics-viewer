#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "The Linux package must be built on Linux." >&2
    exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
spec_file="$project_root/packaging/pyinstaller/napari_compare.spec"
dist_dir="$project_root/dist"
work_dir="$project_root/build/pyinstaller-linux"
artifacts_dir="$project_root/artifacts"
app_name="NapariCompareXeniumMERSCOPE"
app_dir="$dist_dir/$app_name"
version="$(python -c "from importlib.metadata import version; print(version('napari-compare-xenium-merscope'))")"
architecture="$(dpkg --print-architecture)"

rm -rf "$work_dir" "$app_dir"
mkdir -p "$artifacts_dir"
python -m PyInstaller \
    --noconfirm \
    --clean \
    --distpath "$dist_dir" \
    --workpath "$work_dir" \
    "$spec_file"

if [[ ! -x "$app_dir/$app_name" ]]; then
    echo "PyInstaller did not produce $app_dir/$app_name" >&2
    exit 1
fi

package_root="$project_root/build/linux-deb-root"
rm -rf "$package_root"
install -d \
    "$package_root/DEBIAN" \
    "$package_root/opt/napari-compare-xenium-merscope" \
    "$package_root/usr/bin" \
    "$package_root/usr/share/applications" \
    "$package_root/usr/share/icons/hicolor/200x200/apps"
cp -a "$app_dir/." "$package_root/opt/napari-compare-xenium-merscope/"
install -m 0755 \
    "$project_root/packaging/linux/napari-compare-xenium-merscope" \
    "$package_root/usr/bin/napari-compare-xenium-merscope"
install -m 0644 \
    "$project_root/packaging/linux/napari-compare-xenium-merscope.desktop" \
    "$package_root/usr/share/applications/napari-compare-xenium-merscope.desktop"
install -m 0644 \
    "$project_root/src/napari_compare_xenium_merscope/assets/app_icon.png" \
    "$package_root/usr/share/icons/hicolor/200x200/apps/napari-compare-xenium-merscope.png"

installed_size="$(du -sk "$package_root" | cut -f1)"
sed \
    -e "s/@VERSION@/$version/g" \
    -e "s/@ARCHITECTURE@/$architecture/g" \
    -e "s/@INSTALLED_SIZE@/$installed_size/g" \
    "$project_root/packaging/linux/control.in" > "$package_root/DEBIAN/control"

output="$artifacts_dir/napari-compare-xenium-merscope_${version}_${architecture}.deb"
rm -f "$output"
dpkg-deb --root-owner-group --build "$package_root" "$output"

echo "Linux application: $app_dir/$app_name"
echo "Linux package: $output"
