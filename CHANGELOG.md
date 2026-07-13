# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases are cut with `python scripts/release.py {patch|minor|major}`, which
bumps the version, moves the entries below out of *Unreleased*, commits, and
tags. See [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

### Added

- `scripts/release.py` — bumps the version, dates the changelog, commits, and
  tags. Refuses to run on a dirty tree, on failing tests, or with an empty
  Unreleased section.
- Version-controlled git hooks in `.githooks/`. `pre-commit` runs the tests and
  ruff; `post-commit` auto-tags any commit that changes the version, so no
  released version is ever left unreachable.
- `CONTRIBUTING.md`, and ruff as a dev dependency with lint config.

### Changed

- `__version__` is now read from installed package metadata instead of being
  hardcoded, making `pyproject.toml` the single source of truth. The two
  declarations could previously drift apart.
- Modernised typing throughout (`Dict`/`List`/`Optional` → `dict`/`list`/`| None`).

## [4.2.0] - 2026-07-12

First release as an installable package. The project could not run before this:
`hive_orchestrator.py` imported `nodes.execution.*`, which did not exist, so
every entrypoint died on `ImportError` before a single node executed.

### Added

- `src/multi_hive/` package layout, installable with `pip install -e .`, with a
  `multi-hive` console script and `python -m multi_hive` entrypoint.
- `core/ast_utils.py` — imported by both editor nodes and previously missing
  from the repository entirely.
- `core/platform.py` — the Windows/Linux seam for peak-RSS sampling and sandbox
  resource limits.
- `core/console.py` — shared Rich console with a UTF-8 bootstrap.
- A `./workspace` directory for all generated code and artefacts, kept out of
  the source tree.
- pytest suite (17 tests) covering the path boundary, the outline builder, the
  platform shim, and graph construction.

### Fixed

- **Graph could not be built.** Nodes now live at `multi_hive.nodes.execution.*`,
  which is where the orchestrator always expected them.
- **`resource` is Unix-only.** Three modules imported it at module scope, so the
  project raised `ImportError` on Windows before running. Now behind
  `core/platform.py`.
- **Banner emoji crashed the REPL on Windows.** The console defaults to cp1252,
  which cannot encode `🐝`, so startup died with `UnicodeEncodeError`.
- **stdin listener leaked a blocked thread per sprint.** Cancelling the listener
  task did not unblock the thread parked in `readline()`; two sprints exhausted
  the bounded 2-worker pool and the REPL stopped accepting input. Replaced with
  a single shared reader feeding both the REPL and the human gate.
- **Peak RSS reported 0 MB on Windows.** `GetCurrentProcess()` without an
  explicit `restype` truncates the 64-bit pseudo-handle to `-1`.

### Changed

- Generated code now lands in `./workspace/{src,outputs}` rather than `./src`,
  which is the package itself. A task writing to `src/foo.py` would previously
  have written into the source tree.
- All file I/O is explicitly UTF-8. Windows would otherwise default to cp1252
  and raise on the first non-ASCII byte of a traceback.

### Security

- `safe_path()` resolves every model-authored path against the workspace and
  refuses anything escaping `workspace/src` or `workspace/outputs`. Model output
  is untrusted input.

### Known gaps

- Sandbox RLIMIT ceilings (address space, file size, process count) are applied
  on POSIX only. `preexec_fn` requires `fork()`, which Windows lacks; the
  equivalent needs a Job Object. Generated code is still bounded by the
  subprocess timeout and a stripped environment on both platforms.

[Unreleased]: https://github.com/playa/multi_hive/compare/v4.2.0...HEAD
[4.2.0]: https://github.com/playa/multi_hive/releases/tag/v4.2.0
