---
name: create-release
description: >-
  Cut a new versioned release of napari-compare-xenium-merscope: tidy the
  [Unreleased] section of CHANGELOG.md, bump the version with bump-my-version,
  and open a release PR whose merge makes GitHub Actions build the desktop
  packages, tag vX.Y.Z and publish a GitHub Release with the changelog section
  as notes. Use this whenever the user wants to release, ship, publish, or cut
  a new version, bump the version number, or tag a release, even if they just
  say something like "let's put out 0.5.0" or "time to release this". Also use
  it when they ask which version the next release should be, or how to write
  a changelog entry for a change.
---

# Creating a release

Releases are driven by `CHANGELOG.md` and a **release PR**. The full rules are
in [`docs/versioning.md`](../../../docs/versioning.md); read it when anything
here is ambiguous. The tooling is `scripts/ci/changelog.py` (stdlib only) and
`bump-my-version` (config in `pyproject.toml`).

## The rules that shape everything

- **Every version is a release.** There are no untagged patch bumps: patch,
  minor and major releases are all tagged and published. Only release PRs
  change the version.
- **Never push a release tag by hand.** Merging the release PR into `main`
  runs `.github/workflows/package-desktop-apps.yml`, which builds every
  platform from the merge commit and only then tags it `vX.Y.Z` and publishes
  the release. A hand-made tag would skip the builds and the checks.
- **pyproject.toml holds the only copy of the version.** `__version__`,
  `--version`, app bundles and installer names read the package metadata.
- **0.4.0 was never released.** It is in `NEVER_RELEASED` in
  `scripts/ci/changelog.py`; the next release from 0.4.0 is 0.5.0 (`bump
  minor`). Never add a `## [0.4.0]` section.

## Preflight

Stop and explain if any of these fail:

1. `git status --porcelain` is empty (or only holds edits the user wants in
   the release PR).
2. On `main`, up to date: `git fetch` shows no unpulled commits.
3. `python3 scripts/ci/changelog.py versions` and `... validate` pass.
4. `[Unreleased]` in `CHANGELOG.md` has entries. If it is empty, there is
   nothing to release; say so.
5. The target tag does not exist: `git ls-remote --tags origin "v<new>"` is
   empty.

## Choose the version

If the user gave a version, check it against the rules; otherwise propose one
from the `[Unreleased]` entries using the table in `docs/versioning.md`
("Choosing the version"): the largest change decides. Before 1.0.0, breaking
changes raise MINOR and need an `### Upgrading` note. Compare against
`git log v<last>..origin/main --oneline` for user-facing changes that have no
entry, and add entries for them. State the version and why, and let the user
confirm.

## Prepare the release PR

```bash
git switch -c release/v<new>
```

1. Edit `[Unreleased]`: merge duplicates, add `### Highlights` for a minor or
   major release and `### Upgrading` when users must change anything. Keep the
   allowed `###` headings (`Highlights`, `Upgrading`, `Added`, `Changed`,
   `Deprecated`, `Removed`, `Fixed`, `Security`, `Requirements`,
   `Limitations`). Do not add a Downloads section: the workflow appends it.
2. Bump and refresh the links:

   ```bash
   bump-my-version bump <patch|minor|major> --dry-run -v   # preview
   bump-my-version bump <patch|minor|major>                # or: uvx bump-my-version@1.3.0 bump ...
   python3 scripts/ci/changelog.py links --write
   python3 scripts/ci/changelog.py versions
   python3 scripts/ci/changelog.py validate
   python3 scripts/ci/changelog.py notes <new>             # the release notes
   ```

3. Optionally run `python -m pytest -q` (the release builds run it on every
   platform too).
4. Commit as `[release]: v<new>`.

## Open the PR (confirm first)

Pushing the branch and opening the PR is outward-facing, and merging it
publishes a public release with installers. Show the user the version and the
notes, and get explicit confirmation before pushing. Then:

```bash
git push -u origin release/v<new>
gh pr create --title "[release]: v<new>" --body "$(python3 scripts/ci/changelog.py notes <new>)"
```

Do not merge the PR unless the user asks. Tell them that squash-merging it
starts the release, which takes about 90 minutes to build.

## After the merge

- Watch the run: `gh run watch`, or the Actions tab.
- The release appears with `gh release view v<new>` once all platforms build.
- **A build fails:** nothing is tagged. If the failure was flaky (a runner or
  network problem), re-run the workflow on `main` from the Actions tab: it
  rebuilds the commit that set the version. If the code needs a fix, the
  release commit cannot be changed: fix it on `main` in a normal PR (entry under
  `[Unreleased]`) and cut a new patch release whose release PR renames the
  failed version's heading to the new version, folds the fix's entry into it,
  and adds the failed version to `NEVER_RELEASED`.
- **Notes need fixing after publishing:** edit the release on GitHub and fix
  `CHANGELOG.md` in a normal PR labelled `skip-changelog`.

## Changelog entries outside releases

Every PR that touches `src/`, `packaging/`, `pyproject.toml` or
`environment.yml` adds an entry under `## [Unreleased]`, or carries the
`skip-changelog` label when users would notice nothing. Write for users: what
changed, who it affects, and what to do about it.
