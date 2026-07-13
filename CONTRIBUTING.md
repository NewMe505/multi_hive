# Contributing

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate

pip install -e ".[dev]"
git config core.hooksPath .githooks     # once per clone — see "Hooks" below
```

## The loop

```bash
pytest                  # 17 tests, well under a second
ruff check src tests scripts
ruff check --fix src tests scripts
```

## Hooks

Hooks live in `.githooks/` and are version-controlled, so everyone gets the same
ones. Git does not use them until you point it there — once per clone:

```bash
git config core.hooksPath .githooks
```

**`pre-commit`** runs the test suite, and ruff if it is installed. A commit that
breaks the graph never lands. Bypass deliberately with `git commit --no-verify`
when you are checkpointing work in progress.

**`post-commit`** tags any commit that changes the version in `pyproject.toml`
as `v<version>`. This is what makes every released version checkoutable later,
whether the bump came from `scripts/release.py` or from editing the file by
hand.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/). The prefix is what
lets a reader scan history and lets tooling derive a changelog:

```
feat:     a new capability
fix:      a bug fix
perf:     a performance change
refactor: a change that alters no behaviour
docs:     documentation only
test:     tests only
chore:    tooling, deps, release plumbing
```

Write the body for someone who will read it in a year with no memory of today.
Say what was wrong and why the fix is the fix — not what the diff already shows.

## Versioning

[Semantic Versioning](https://semver.org). `pyproject.toml` is the single source
of truth; `multi_hive.__version__` reads it back from installed metadata, so the
two cannot drift.

Record every user-visible change under `## [Unreleased]` in `CHANGELOG.md` as
you make it, not at release time when you have forgotten the details.

Cut a release:

```bash
python scripts/release.py patch      # 4.2.0 -> 4.2.1   bug fixes
python scripts/release.py minor      # 4.2.0 -> 4.3.0   new features
python scripts/release.py major      # 4.2.0 -> 5.0.0   breaking changes

python scripts/release.py patch --dry-run    # see the plan, change nothing
```

It refuses to run on a dirty tree, refuses if the tests fail, and refuses if
`## [Unreleased]` is empty. Then it bumps the version, dates the changelog
section, commits as `chore(release): vX.Y.Z`, and tags.

```bash
git push && git push --tags
```

## Line endings

`.gitattributes` pins everything to LF, in the repository and in the working
tree, on both platforms. This project runs on Windows and Linux; without that
pin, a file authored on one lands on the other as an all-lines-changed diff.

## Where things live

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
