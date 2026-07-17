# Releasing

Releases are produced by the **Package desktop applications** workflow
([`.github/workflows/package-desktop-apps.yml`](.github/workflows/package-desktop-apps.yml)).
The workflow builds the macOS (arm64 + x86_64), Windows, and Linux packages,
generates `SHA256SUMS.txt`, and publishes a GitHub Release with auto-generated
notes and all artifacts attached.

## The versioning convention

The version lives in **one place** — the `version` field in
[`pyproject.toml`](pyproject.toml). Everything else (app bundle metadata,
installer names, artifact filenames) reads it at build time, so this is the only
number you ever edit.

A **git tag is what triggers a release** — not the version number itself. This
lets small changes carry a version without producing a downloadable build:

| Change                     | What you do                                  | Release? |
| -------------------------- | -------------------------------------------- | -------- |
| Patch, e.g. `0.1.0→0.1.1`  | Bump `pyproject.toml`, commit. **No tag.**   | No       |
| Minor, e.g. `0.1.x→0.2.0`  | Bump `pyproject.toml`, commit, **push tag**. | Yes      |
| Major, e.g. `0.x→1.0.0`    | Bump `pyproject.toml`, commit, **push tag**. | Yes      |

Patch bumps still increment the version and live in git history; they just never
get tagged, so no release is built. When you decide a set of changes is worth a
release, that becomes a minor (or major) bump and you tag it.

## Cutting a release

1. Bump `version` in [`pyproject.toml`](pyproject.toml) (e.g. `0.1.0` → `0.2.0`).
2. Commit it:
   ```bash
   git add pyproject.toml
   git commit -m "[release] v0.2.0"
   git push
   ```
3. Tag the commit and push the tag:
   ```bash
   git tag -a v0.2.0 -m "v0.2.0"
   git push origin v0.2.0
   ```

Pushing the tag starts the workflow. The first job, **Verify tag matches package
version**, fails fast if the tag (`v0.2.0`) and `pyproject.toml` (`0.2.0`)
disagree — so a mismatched tag stops before the ~90-minute platform builds run.
When the builds finish, the release is published automatically.

> The tag name must be `v` + the exact pyproject version (`v0.2.0`, not
> `0.2.0`).

### Test builds without releasing

Use **Run workflow** (the `workflow_dispatch` trigger) on the Actions tab to
build all platforms and download the artifacts *without* creating a release.
This is the old manual flow, kept for verifying a build before you tag it.

## Release notes

The workflow uses GitHub's built-in note generator
(`generate_release_notes: true`). [`.github/release.yml`](.github/release.yml)
groups entries into **Features / Fixes / Performance / Maintenance & Docs** based
on the **labels of merged pull requests**.

**Caveat:** grouping only works for changes that arrive via labelled PRs. This
repo has been using direct commits to `main` (e.g. `[bugfix] …`, `[feature] …`).
For those, GitHub still generates notes but as a **flat list of commits** — the
categories will be empty. To get the grouped changelog, route changes through
PRs and apply labels (`feature`, `bug`, `performance`, `documentation`, …). The
release is fully functional either way; only the formatting of the notes
differs.

## What gets attached to a release

- `Napari-Compare-Xenium-MERSCOPE-<version>-macOS-arm64.dmg`
- `Napari-Compare-Xenium-MERSCOPE-<version>-macOS-x86_64.dmg`
- `Napari-Compare-Xenium-MERSCOPE-<version>-Windows-x86_64-Setup.exe`
- `napari-compare-xenium-merscope_<version>_<arch>.deb`
- `SHA256SUMS.txt` — SHA-256 of each of the above, by basename.

Verify a download with:

```bash
sha256sum -c SHA256SUMS.txt --ignore-missing
```
