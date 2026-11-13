# Releasing

Releases are made from [CHANGELOG.md](CHANGELOG.md) by merging a release PR.
The full process, the rules for choosing the version and the checks CI runs are
in [docs/versioning.md](docs/versioning.md). In short:

```bash
git switch main && git pull && git switch -c release/vX.Y.Z
# tidy the [Unreleased] entries in CHANGELOG.md, then:
bump-my-version bump minor          # or major / patch
python3 scripts/ci/changelog.py links --write
python3 scripts/ci/changelog.py versions && python3 scripts/ci/changelog.py validate
git commit -am "[release]: vX.Y.Z"  # open a PR with this title, squash-merge it
```

Merging the release PR into `main` runs the
[Package desktop applications](.github/workflows/package-desktop-apps.yml)
workflow. It builds and smoke-tests every platform from the merge commit, then
tags it `vX.Y.Z` and publishes a GitHub Release whose notes are the version's
CHANGELOG section. Do not push release tags by hand.

## What gets attached to a release

- `Napari-Compare-Xenium-MERSCOPE-<version>-macOS-arm64.dmg`
- `Napari-Compare-Xenium-MERSCOPE-<version>-macOS-x86_64.dmg`
- `Napari-Compare-Xenium-MERSCOPE-<version>-Windows-x86_64-Setup.exe`
- `napari-compare-xenium-merscope_<version>_<arch>.deb`
- `SHA256SUMS.txt`: the SHA-256 of each of the above, by basename.

Verify a download with:

```bash
sha256sum -c SHA256SUMS.txt --ignore-missing
```
