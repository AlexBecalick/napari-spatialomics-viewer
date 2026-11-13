# Versioning and releases

The viewer follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
A release is numbered `MAJOR.MINOR.PATCH`, tagged `vMAJOR.MINOR.PATCH` on `main`
and published as a [GitHub release](https://github.com/AlexBecalick/napari-spatialomics-viewer/releases)
with the desktop installers attached and its section of
[CHANGELOG.md](../CHANGELOG.md) as the notes.

- **MAJOR** goes up when something that worked on the previous release needs
  the user to change something: a command-line flag, a file the viewer writes,
  or a kind of store it opens was removed or renamed, or became incompatible.
- **MINOR** goes up for new features and for changed behaviour that a setting
  can switch back.
- **PATCH** goes up for bug fixes, even when they change what the viewer shows
  or writes (the entry says which), and for changes users do not see.

**Before 1.0.0** the viewer is in initial development, as SemVer allows for
`0.y.z`: a breaking change raises MINOR (not MAJOR) and is listed under
`### Upgrading` in its entry. Features also raise MINOR and fixes raise PATCH.
Releasing 1.0.0 commits to the major/minor/patch rules above.

## Which version am I running?

| Where | What it shows |
| --- | --- |
| `napari-compare-xenium-merscope --version` | The installed viewer release. |
| Installer and app file names | `Napari-Compare-Xenium-MERSCOPE-<version>-…`, `napari-compare-xenium-merscope_<version>_<arch>.deb`. |
| macOS **Get Info** / app bundle | `CFBundleShortVersionString` is the release. |
| Manual-registration drafts and results | The payload's `software_version`. |

The version is read from the installed package metadata, which comes from
`pyproject.toml`; that is the only file that holds the number. An editable
install reports the version it was installed with, so re-run
`pip install -e .` after pulling a release to update what `--version` shows.

Between releases, `main` keeps the last release's version (or a version listed
as never released, below), so a build from `main` is identified by its commit,
not by its version.

## What the version covers

The version applies to what users run, open or keep:

- **The command line**: flags, their accepted values and defaults
  (`napari-compare-xenium-merscope --help`).
- **Stores the viewer opens**: the platforms (MERSCOPE, Xenium, Atera), the
  MerXen schema versions and layouts it accepts (paired v2 materialised stores,
  manual-registration schema version 1, HITL `roi_context.json`), and the
  SpatialData/Zarr store format, which follows the pinned `spatialdata` and
  `zarr` versions.
- **Files the viewer writes**: annotation exports (cortical depth, distance
  from object, object polygons), cell-group CSV exports, HITL region GeoJSON
  (`merxen_hitl_regions_version`) and annotation packages, and
  manual-registration drafts and results. Their names, formats and documented
  fields.
- **Saved local settings**: recently viewed datasets, saved rotations and
  remembered environments, which a release should keep reading.
- **Installers**: the supported operating systems and architectures.

Not covered: the Python modules (there is no public Python API), the layout of
the viewer's caches outside the store (rebuilt when stale), tests, scripts under
`scripts/`, and the exact look of the panels. They can change in any release;
their changes still get a changelog entry when users would notice.

## Choosing the version

Choose the version from the [Unreleased] entries: the largest change decides.
Before 1.0.0, read "major" below as "minor".

| Change | Version |
| --- | --- |
| Removing or renaming a flag, or changing its meaning so an old invocation fails or does something else | major |
| Refusing a store layout or MerXen schema version that the previous release opened | major |
| Changing a written file's format or fields so that existing readers (MerXen included) break | major |
| Dropping an operating system, architecture or Python version | major |
| A new tab, flag, export, platform or store layout that existing use does not have to touch | minor |
| A changed default when a setting restores the old behaviour. The entry names the setting. | minor |
| Raising a pinned storage dependency (`spatialdata`, `zarr`, `anndata`) when older stores still open | minor |
| Deprecating something: it still works, warns, and is removed in the next major | minor |
| A bug fix. The entry says what changes on screen or in written files. | patch |
| Faster or lighter loading with the same results, packaging fixes, documentation | patch |

Prefer deprecating first: keep an old flag or format working with a warning
for at least one minor release, then remove it in the next major.

**Never-released versions.** `0.4.0` was set in `pyproject.toml` before this
process existed and was never tagged. It is listed in `NEVER_RELEASED` in
[`scripts/ci/changelog.py`](../scripts/ci/changelog.py), which lets `main`
carry it until the next release (0.5.0 by a minor bump) and refuses a
changelog section for it. Builds and saved files stamped 0.4.0 come from `main`
between 0.3.0 and 0.5.0 and do not identify a commit.

## Changelog entries

[CHANGELOG.md](../CHANGELOG.md) follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

- **Every PR adds an entry.** A pull request that changes what users run adds
  its entry under `## [Unreleased]` in the same PR. The `Changelog` CI job
  fails a PR that changes `src/`, `packaging/`, `pyproject.toml` or
  `environment.yml` without touching `CHANGELOG.md`. Label the PR
  `skip-changelog` when users would notice nothing, such as a refactor with
  identical behaviour or a test-only change.
- **Change types.** File the entry under one of these `###` headings: `Added`,
  `Changed`, `Deprecated`, `Removed`, `Fixed` or `Security`. A release can also
  have `Highlights` (a short summary at the top), `Upgrading` (what users must
  do), `Requirements` (what stores or MerXen versions it needs) and
  `Limitations` (known gaps). A `####` heading can group the bullets of a
  larger change.
- **Write for users.** Say what changed, who it affects and what to do about
  it. Give the setting that restores the old behaviour, and link the docs.

The `Changelog` job also checks that the version agrees across
`pyproject.toml`'s `[project]` table and its bump-my-version config, and that
`CHANGELOG.md` is well formed, its compare links included. Run the same checks
locally:

```bash
python3 scripts/ci/changelog.py versions
python3 scripts/ci/changelog.py validate
python3 scripts/ci/changelog.py check-pr --base origin/main
```

## Making a release

1. **Branch from `main`.**

   ```bash
   git switch main && git pull
   git switch -c release/v0.5.0
   ```

2. **Edit the [Unreleased] entries.** Merge entries that duplicate each other.
   Add `### Highlights` for a minor or major release, and `### Upgrading` when
   users must change anything. Choose the version as described in
   [Choosing the version](#choosing-the-version).

3. **Set the version.** `bump-my-version` writes it into `pyproject.toml` and
   renames `## [Unreleased]` to `## [0.5.0] - <today>` above a new empty
   `## [Unreleased]`; `links --write` then updates the compare links at the
   bottom of `CHANGELOG.md`. The dev extra installs `bump-my-version` (or use
   `uvx bump-my-version@1.3.0`). Preview it first with `--dry-run -v`.

   ```bash
   bump-my-version bump minor   # or major / patch
   python3 scripts/ci/changelog.py links --write
   python3 scripts/ci/changelog.py versions
   python3 scripts/ci/changelog.py validate
   python3 scripts/ci/changelog.py notes 0.5.0   # the GitHub release notes
   ```

4. **Open the release PR.** Commit as `[release]: v0.5.0` and open a PR with
   that title and the release notes as its body. Once its checks pass,
   squash-merge it. If another PR reached `main` first, merge `main` into the
   release branch and move that PR's entries into the release section. If the
   merge happens on a later day, fix the date in the heading first.

5. **Check the release.** The
   [Package desktop applications](../.github/workflows/package-desktop-apps.yml)
   workflow runs when the merge reaches `main`. It checks that the versions
   agree, that the version is newer than every earlier `v*` tag and that
   `CHANGELOG.md` has its section. It then builds and smoke-tests the macOS,
   Windows and Linux installers from the merge commit (about 90 minutes), tags
   that commit `v0.5.0` and publishes the GitHub release with the section as
   its notes, the installers and `SHA256SUMS.txt`:

   ```bash
   gh run watch
   gh release view v0.5.0
   ```

   Nothing is tagged until every platform has built, so a failed build leaves
   no tag behind. For a flaky failure (a runner or network problem), re-run
   the workflow on `main` from the Actions tab (`workflow_dispatch`): it builds
   and tags the commit that set the version, not the newest `main`. When the
   code needs a fix, fix it on `main` in a normal PR and make a new patch
   release whose release PR renames the failed version's heading to the new
   version (and adds the fix's entry to it), so the changelog lists only
   tagged releases. Add the failed version to `NEVER_RELEASED`.

A push to `main` whose version is already tagged, or that has no changelog
section (a never-released version), builds nothing.

**Test builds.** Run the workflow from the Actions tab on any branch to build
and smoke-test every platform and download the installers as workflow
artifacts. Off `main`, or when the version is already released, it publishes
nothing.

**Merge method.** Squash-merge release PRs as usual. The workflow tags the
first-parent commit on `main` that set the version, so a merge commit also
works, but a rebase merge would tag the bump commit only.

**Pre-releases** (`0.5.0-rc.1`) are not used yet: the tooling accepts only
`X.Y.Z`. To try unreleased work, use a test build.

**Patch releases of an older version** have no automation. Branch from the old
tag, bump, build with a test run of the workflow on that branch, then create
the tag and release by hand with `gh release create`.

## Repository setup

These GitHub settings are needed once, by a repository admin:

- **The `skip-changelog` label**, which exempts a PR from the changelog check:
  `gh label create skip-changelog --description "PR needs no CHANGELOG entry"`.
- **Workflow permissions** that let the release job create tags and releases.
  It asks for `contents: write` itself.
- **Optional:** branch protection on `main` that requires the `Changelog`
  check, so that it blocks a merge instead of only warning.
