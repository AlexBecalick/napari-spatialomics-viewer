---
name: create-release
description: >-
  Cut a new versioned release of napari-compare-xenium-merscope: bump the
  version in pyproject.toml, commit, tag, and push so the GitHub Actions release
  workflow builds the desktop packages and publishes a GitHub Release. Use this
  whenever the user wants to release, ship, publish, or cut a new version, bump
  the version number, or tag a release — even if they just say something like
  "let's put out 0.3.0" or "time to release this". Also use it for version-only
  bumps that should NOT produce a release (patch bumps), since choosing between
  those is the core of this repo's release convention.
---

# Creating a release

This repository releases via **git tags**. The version is single-sourced in
`pyproject.toml`; everything else (app bundle metadata, installer names, artifact
filenames) reads it at build time. Pushing a `v*` tag triggers the
[`Package desktop applications`](../../../.github/workflows/package-desktop-apps.yml)
workflow, which builds macOS/Windows/Linux packages, writes `SHA256SUMS.txt`, and
publishes a GitHub Release with auto-generated notes. The full convention is in
[`RELEASING.md`](../../../RELEASING.md) — read it if anything here is ambiguous.

## The one rule that shapes everything

**The tag — not the version number — is the release switch.** This lets small
changes carry a version without producing a downloadable build:

- **Patch bump** (`0.1.0 → 0.1.1`): bump `pyproject.toml` and commit. **No tag,
  no release.** The version still lives in git history.
- **Minor/major bump** (`0.1.x → 0.2.0`, `0.x → 1.0.0`): bump, commit, **and push
  a tag**. That tag builds and publishes a release.

So before doing anything, be clear on which the user wants. If they said
"release", "ship", "publish", or "cut a version", they want a **tagged release**
(minor/major). If they just want to record a small fix without a build, that's a
**patch bump, no tag**. When it's genuinely unclear, ask — the difference is
whether a public release gets built.

## Before you touch anything: preflight

Run these checks and stop if something is off, explaining what you found rather
than pushing through it:

1. **Confirm the current version**: read `version` from `pyproject.toml`.
2. **Clean working tree**: `git status --porcelain` should be empty. Uncommitted
   changes mean the release wouldn't capture what the user expects.
3. **On the main branch and up to date**: `git branch --show-current` is `main`,
   and `git fetch` then confirm no unpulled commits. Tagging a stale commit
   releases stale code.
4. **The target tag doesn't already exist**: `git tag -l "v<new>"` must be empty,
   and check the remote too (`git ls-remote --tags origin "v<new>"`). Tags are
   effectively immutable once released — never overwrite one.

## Decide the new version

If the user gave an explicit version, use it. Otherwise infer the bump type from
what changed since the last tag (`git log <last-tag>..HEAD --oneline`) and
propose one, following semantic versioning:

- **patch** — bug fixes / internal changes only
- **minor** — new user-facing features, backwards compatible
- **major** — breaking changes (note: pre-1.0, breaking changes conventionally go
  in the minor slot)

State the proposed version and why, and let the user confirm before proceeding.

## Do the bump

Edit the single `version = "..."` line in `pyproject.toml`. Optionally run the
test suite locally first (`python -m pytest -q`) — CI runs it too, but catching a
failure now avoids a wasted tag. Then commit:

```bash
git add pyproject.toml
git commit -m "[release] v<new>"
```

**If this is a patch bump with no release:** push the commit and stop here.

```bash
git push
```

Report that the version is bumped, no release was created (by design), and that a
release will happen on the next minor/major tag.

## Tag and publish (minor/major only)

This is the point of no return — pushing a tag builds and publishes a **public
GitHub Release**. Because that is outward-facing and hard to reverse, **show the
user exactly what will happen and get explicit confirmation before pushing the
tag**: the version, that it will trigger the workflow, and that it will create a
public release with attached installers.

After confirmation:

```bash
git push                                 # push the version-bump commit first
git tag -a v<new> -m "v<new>"            # annotated tag
git push origin v<new>                   # this triggers the release
```

The tag name must be `v` + the exact `pyproject.toml` version (`v0.2.0`, not
`0.2.0`). The workflow's first job verifies the tag matches `pyproject.toml` and
fails fast if not, so a mismatch is caught in seconds rather than after the long
platform builds — but get it right the first time.

## After pushing

Point the user at the run and the resulting release:

- Watch the build: the **Actions** tab on GitHub, or `gh run watch` if the GitHub
  CLI is available.
- The release appears at `https://github.com/AlexBecalick/napari-spatialomics-viewer/releases`
  once all platform builds finish (~up to 90 min).

Do **not** try to draft the release, upload artifacts, or write checksums by hand
— the workflow does all of that. Your job ends once the tag is pushed and the run
has started.

## If something goes wrong

- **Tag pushed with a wrong/mismatched version**: the `verify-version` job will
  fail before building. Delete the bad tag locally and remotely
  (`git tag -d v<bad>` and `git push origin :refs/tags/v<bad>`), fix
  `pyproject.toml`, and start over. Only do this for a tag whose release hasn't
  been published yet.
- **A build job fails**: the release isn't published (the release job needs all
  builds to succeed). Fix the underlying issue on `main`, then cut a new patch or
  minor version — don't reuse the tag.
