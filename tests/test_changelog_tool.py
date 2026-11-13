"""Checks of scripts/ci/changelog.py, the release-version and CHANGELOG tool."""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("tomllib")

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("changelog_tool", ROOT / "scripts/ci/changelog.py")
changelog = importlib.util.module_from_spec(_spec)
sys.modules["changelog_tool"] = changelog
_spec.loader.exec_module(changelog)

URL = changelog.REPOSITORY_URL


def _changelog(sections: str, links: str | None = None) -> str:
    body = "# Changelog\n\n## [Unreleased]\n\n" + textwrap.dedent(sections).strip("\n") + "\n"
    if links is None:
        links = "\n".join(changelog.expected_links(body))
    return body + "\n" + links + "\n"


GOOD = _changelog(
    """
    ## [1.1.0] - 2026-02-01

    ### Added

    - A thing.

    ## [1.0.0] - 2026-01-01

    ### Fixed

    - Another thing.
    """
)


def _repo(tmp_path: Path, version: str, text: str = GOOD) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nversion = "{version}"\n\n[tool.bumpversion]\ncurrent_version = "{version}"\n',
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(text, encoding="utf-8")
    return tmp_path


def test_the_repository_changelog_and_version_are_consistent():
    version, problems = changelog.check_versions()
    assert problems == []
    assert version is not None
    assert changelog.validate((ROOT / "CHANGELOG.md").read_text(encoding="utf-8")) == []


def test_a_well_formed_changelog_validates_and_yields_notes():
    assert changelog.validate(GOOD) == []
    assert changelog.notes(GOOD, "1.0.0") == "### Fixed\n\n- Another thing."
    with pytest.raises(KeyError):
        changelog.notes(GOOD, "9.9.9")


def test_expected_links_compare_each_release_with_the_one_below():
    assert changelog.expected_links(GOOD) == [
        f"[Unreleased]: {URL}/compare/v1.1.0...HEAD",
        f"[1.1.0]: {URL}/compare/v1.0.0...v1.1.0",
        f"[1.0.0]: {URL}/releases/tag/v1.0.0",
    ]


def test_write_links_repairs_a_stale_link_block():
    stale = _changelog(
        """
        ## [1.1.0] - 2026-02-01

        - A thing.

        ## [1.0.0] - 2026-01-01

        - Another thing.
        """,
        links=f"[Unreleased]: {URL}/compare/v1.0.0...HEAD\n[1.0.0]: {URL}/releases/tag/v1.0.0",
    )
    assert any("compare links" in problem for problem in changelog.validate(stale))
    assert not any("compare links" in problem for problem in changelog.validate(changelog.write_links(stale)))


@pytest.mark.parametrize(
    ("sections", "expected"),
    [
        ("## [1.0.0] - 2026-01-01\n\n- x\n\n## [1.1.0] - 2026-02-01\n\n- y", "not older"),
        ("## [1.0.0] - 2026-01-01\n\n### Misc\n\n- x", "change type"),
        ("## [1.0.0] - 2026-01-01\n", "is empty"),
        ("## 1.0.0\n\n- x", "neither"),
        ("## [0.4.0] - 2026-01-01\n\n- x", "never released"),
    ],
)
def test_validate_reports_layout_problems(sections, expected):
    problems = changelog.validate(_changelog(sections))
    assert any(expected in problem for problem in problems), problems


def test_versions_must_agree_and_match_the_newest_release(tmp_path):
    assert changelog.check_versions(_repo(tmp_path, "1.1.0")) == ("1.1.0", [])
    _, problems = changelog.check_versions(_repo(tmp_path, "1.2.0"))
    assert any("not the newest" in problem for problem in problems)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "1.1.0"\n\n[tool.bumpversion]\ncurrent_version = "1.0.0"\n', encoding="utf-8"
    )
    version, problems = changelog.check_versions(tmp_path)
    assert version is None and any("differs" in problem for problem in problems)


def test_a_never_released_version_is_accepted_only_above_the_newest_release(tmp_path):
    old = _changelog("## [0.3.0] - 2026-01-01\n\n- x")
    assert changelog.check_versions(_repo(tmp_path, "0.4.0", old)) == ("0.4.0", [])
    newer = _changelog("## [0.5.0] - 2026-02-01\n\n- y\n\n## [0.3.0] - 2026-01-01\n\n- x")
    _, problems = changelog.check_versions(_repo(tmp_path, "0.4.0", newer))
    assert any("never released" in problem for problem in problems)


def test_versions_must_be_newer_than_existing_tags(tmp_path):
    root = _repo(tmp_path, "1.1.0")
    assert changelog.check_versions(root, tags=["v1.0.0", "v1.1.0", "not-a-version"])[1] == []
    _, problems = changelog.check_versions(root, tags=["v1.2.0"])
    assert any("v1.2.0" in problem for problem in problems)


def test_release_status_releases_an_untagged_changelog_version_only():
    assert changelog.release_status("1.1.0", GOOD, ["v1.0.0"]) == "release"
    assert changelog.release_status("1.1.0", GOOD, ["v1.0.0", "v1.1.0"]) == "skip"
    assert changelog.release_status("0.4.0", GOOD, []) == "skip"


def test_relative_links_become_absolute_in_release_notes():
    text = "See [docs](docs/a.md), [site](https://x.org), [here](#anchor) and [root](/r)."
    assert changelog.absolute_links(text, "https://base/blob/v1/") == (
        "See [docs](https://base/blob/v1/docs/a.md), [site](https://x.org), [here](#anchor) and [root](/r)."
    )


def test_user_facing_paths_need_an_entry():
    assert changelog.needs_entry(
        ["src/napari_compare_xenium_merscope/viewer.py", "tests/test_x.py", "packaging/linux/build_linux.sh",
         "pyproject.toml", "README.md", "docs/a.md"]
    ) == ["src/napari_compare_xenium_merscope/viewer.py", "packaging/linux/build_linux.sh", "pyproject.toml"]
