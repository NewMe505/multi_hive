"""
release.py — cut a release.

    python scripts/release.py patch    4.2.0 -> 4.2.1   bug fixes
    python scripts/release.py minor    4.2.0 -> 4.3.0   new features, back-compatible
    python scripts/release.py major    4.2.0 -> 5.0.0   breaking changes

    python scripts/release.py patch --dry-run    show what would happen

What it does, in order:
  1. refuses to run on a dirty tree (a release must be reproducible from a commit)
  2. runs the test suite (never tag something that doesn't pass)
  3. bumps the version in pyproject.toml — the single source of truth
  4. moves everything under "## [Unreleased]" in CHANGELOG.md into the new
     version's section, stamped with today's date
  5. commits as `chore(release): v<version>`
  6. creates an annotated tag `v<version>`

The tag is what makes a version recoverable: `git checkout v4.2.0` gets you
exactly the tree that was tested and shipped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"

VERSION_RE = re.compile(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', re.MULTILINE)


def fail(message: str) -> NoReturn:
    print(f"release: {message}", file=sys.stderr)
    raise SystemExit(1)


def git(*args: str, capture: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=capture,
        text=True,
    )
    if result.returncode != 0:
        fail(f"`git {' '.join(args)}` failed:\n{result.stderr or result.stdout}")
    return (result.stdout or "").strip()


def current_version() -> tuple[int, int, int]:
    match = VERSION_RE.search(PYPROJECT.read_text(encoding="utf-8"))
    if not match:
        fail("no `version = \"X.Y.Z\"` found in pyproject.toml")
    return tuple(int(g) for g in match.groups())  # type: ignore[return-value]


def bump(version: tuple[int, int, int], part: str) -> str:
    major, minor, patch = version
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def write_version(new: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    PYPROJECT.write_text(
        VERSION_RE.sub(f'version = "{new}"', text, count=1),
        encoding="utf-8",
    )


def update_changelog(new: str) -> None:
    """
    Promotes the Unreleased section to a dated release section.

    Refuses on an empty Unreleased block: a release with no changelog entry is a
    release nobody can understand six months later.
    """
    text = CHANGELOG.read_text(encoding="utf-8")

    if "## [Unreleased]" not in text:
        fail("CHANGELOG.md has no `## [Unreleased]` section")

    body_start = text.index("## [Unreleased]") + len("## [Unreleased]")
    next_section = text.find("\n## [", body_start)
    body = text[body_start : next_section if next_section != -1 else len(text)]

    if not body.strip():
        fail(
            "nothing under `## [Unreleased]` in CHANGELOG.md — "
            "describe the change before releasing it"
        )

    today = dt.date.today().isoformat()
    text = text.replace(
        "## [Unreleased]",
        f"## [Unreleased]\n\n## [{new}] - {today}",
        1,
    )
    CHANGELOG.write_text(text, encoding="utf-8")


def run_tests() -> None:
    python = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    interpreter = str(python) if python.exists() else sys.executable

    print("release: running tests")
    result = subprocess.run([interpreter, "-m", "pytest", "-q"], cwd=ROOT)
    if result.returncode != 0:
        fail("tests failed — not releasing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut a release.")
    parser.add_argument("part", choices=["major", "minor", "patch"])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned version and exit without touching anything",
    )
    args = parser.parse_args()

    old = current_version()
    new = bump(old, args.part)
    old_str = ".".join(str(n) for n in old)

    if args.dry_run:
        print(f"release: {old_str} -> {new} ({args.part}) [dry run, nothing changed]")
        return

    if git("status", "--porcelain"):
        fail("working tree is dirty — commit or stash first")

    if git("tag", "--list", f"v{new}"):
        fail(f"tag v{new} already exists")

    run_tests()

    write_version(new)
    update_changelog(new)

    git("add", "pyproject.toml", "CHANGELOG.md")
    git("commit", "-m", f"chore(release): v{new}")

    # The post-commit hook tags any commit that changes the version, so by the
    # time we get here the tag usually already exists — the hook fires during
    # the commit above. Tagging again is a hard error, so check.
    #
    # Both paths are wanted: the hook covers a version bumped by hand, and this
    # covers a repo whose hooks are not installed (core.hooksPath is per-clone).
    if git("tag", "--list", f"v{new}"):
        print(f"\nrelease: {old_str} -> {new}")
        print(f"release: committed; v{new} tagged by the post-commit hook")
    else:
        git("tag", "-a", f"v{new}", "-m", f"v{new}")
        print(f"\nrelease: {old_str} -> {new}")
        print(f"release: committed and tagged v{new}")

    print("release: refresh the installed metadata with  pip install -e '.[dev]'")
    print("release: push with  git push && git push --tags")


if __name__ == "__main__":
    main()
