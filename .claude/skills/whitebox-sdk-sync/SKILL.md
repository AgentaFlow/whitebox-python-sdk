---
name: whitebox-sdk-sync
description: Ports fixes/features from the WhiteBoxXAI application monorepo's canonical `sdk/whiteboxxai` source into this standalone package, then verifies and prepares a release. TRIGGER — read BEFORE: syncing, updating, or releasing a new version of this SDK from `whitebox-xai-azure/sdk`; checking whether this repo has drifted behind the monorepo's SDK source; preparing a CHANGELOG entry or version bump for a sync-driven release. SKIP when the task doesn't involve the monorepo as a source (e.g. a standalone bug fix, doc typo, or feature written directly in this repo with no monorepo counterpart).
user-invocable: true
---

# Syncing whitebox-python-sdk from whitebox-xai-azure/sdk

This repo (`whitebox-xai-sdk` on PyPI, importable as `whiteboxxai`) is a hand-maintained port of
the SDK code that lives canonically inside the main application monorepo, at
`whitebox-xai-azure/sdk/whiteboxxai` (local path on this machine:
`c:\github\whitebox-xai-azure\sdk\whiteboxxai` — confirm with the user if it's cloned elsewhere).
The monorepo is the source of truth; this repo is the distributable package. Periodically, fixes
and features land in the monorepo's copy and need porting here before a release.

This file's own "Last synced state" section below is the durable record of where the last sync
left off — keep it current (last step of the process below), the same way
[[whitebox-azure-waf]] keeps `WAF_ALIGNMENT.md` current rather than letting it drift into a stale
snapshot.

## Last synced state (keep this current)

- **Monorepo commit ported through:** `c1d4aaa0` (2026-07-20, "Align integration logging APIs and tests")
- **Also ported:** `34ba0ee2` (2026-07-17, langchain_core import fallback)
- **Standalone version released:** `1.1.0`
- **Date:** 2026-07-30
- **PR:** [whitebox-python-sdk#7](https://github.com/AgentaFlow/whitebox-python-sdk/pull/7)

## Process

1. **Find what's new.** In the monorepo, `git log --oneline <last-synced-sha>..HEAD -- sdk/whiteboxxai`
   to list commits since the last sync. Read each commit's actual diff
   (`git show <sha> -- sdk/whiteboxxai`) rather than trusting commit messages — they're not always
   accurate (one past commit here was titled "mypy SQLAlchemy false positives" but its real content
   was pure black reformatting).

2. **Classify each change: functional or noise.** A raw `git diff --no-index` between the
   monorepo's `sdk/whiteboxxai` and this repo's `whiteboxxai/` will show far more files differing
   than actually changed — most of it is black/isort line-wrap noise, because this repo gets its
   own formatting pass and the two copies drift stylistically even when semantically identical.
   Worse, the monorepo copy accumulates genuine dead code between syncs (unused imports, stray
   leftover edits) that predates the commits you're porting. Before porting anything from a diff:
   - Confirm the change is actually exercised by `git show <sha> -- <path>` for a specific commit,
     not just "this file differs somehow."
   - For any new import, grep for its actual use in the file. This repo's `.flake8` ignores E501
     but **not** F401 — an unused import ported verbatim will fail the blocking `flake8` CI job.
     Past sync found unused `Union`/`Callable`/`json`/`numpy as np`/`List`/`import time`/
     `from datetime import datetime` sitting in the monorepo copy across several files; none of
     it was part of the commits actually being ported, and none of it should be copied.
   - Never bulk-overwrite a whole file from the monorepo copy (that's how the v1.0.0 initial port
     worked, but it reintroduces formatting churn and dead code). Hand-port only the specific
     functional hunks into this repo's already-formatted files.

3. **Port the functional changes**, then run `black --check`, `isort --check-only`, and
   `flake8 <file> --max-line-length=120` on each touched file immediately. If `black --check`
   wants changes, run `black <file>` to auto-fix rather than hand-wrapping lines — this repo's
   configured line length may not match what you'd guess.

4. **Add regression tests when they're cheap and the fix is otherwise invisible to CI.** Skip
   integration modules that have no existing test scaffolding and whose real package isn't
   installed locally (e.g. `xgboost`/`tensorflow`/`transformers` weren't available in the dev
   environment during the 1.1.0 sync) — for those, rely on `python -c "import ast; ast.parse(...)"`
   as a syntax check plus careful manual verification against the monorepo diff, and treat adding
   real test coverage as a separate follow-up rather than blocking the release on it.

5. **Run the full suite and compare to a baseline**, not just "tests pass":
   `pytest tests/unit -v`, `tests/integration -v`, `tests/e2e -v`, plus a full-repo
   `black --check --diff .` / `isort --check-only --diff .` /
   `flake8 whiteboxxai/ tests/ examples/ --max-line-length=120`. Record pass counts before you
   start so you can confirm nothing regressed, not just that nothing errored.

6. **Check docs for hand-written signatures that drifted.** `docs/api-reference.md` documents
   some method signatures by hand (not via mkdocstrings autodoc) — grep it for any method name you
   changed. If anything under `docs/**` changed, run `mkdocs build --strict` (this repo's
   `docs.yml` CI runs it in blocking mode, unlike its non-blocking `markdownlint ... || true` step).

7. **Update `CHANGELOG.md`** in Keep a Changelog format (`### Fixed` / `### Added` / `### Changed`
   as needed), written after verification is green so it describes what actually shipped. Cite the
   monorepo commit SHA(s) ported in the entry or commit message so the next sync doesn't have to
   redo the archaeology this skill file's "Last synced state" section is meant to avoid. Only touch
   the reference-link footer if it's already broken (check `git tag -l` against the existing links
   — this repo's tags are **unprefixed**, `1.0.0` not `v1.0.0`; a past footer pointed at
   nonexistent `v`-prefixed tags).

8. **Version bump: ask, don't auto-derive.** Semver "fix vs. feature" doesn't map cleanly onto
   this project's actual release cadence — the 1.1.0 release bundled several bug fixes under a
   **minor** bump by explicit user direction, not because pure semver would have called for a
   minor bump. Confirm the target version number with the user rather than deciding unilaterally.

9. **Commit only the specific files touched** (never `git add -A`) on a dedicated branch.

10. **Stop before pushing or opening a PR.** Show the diff and ask first — pushing and opening a
    PR are visible, semi-hard-to-reverse actions that warrant an explicit go-ahead each time, even
    if a prior sync was approved. Cutting the actual GitHub Release (which triggers the PyPI
    publish via OIDC trusted publishing in `publish.yml`) is a separate, later, explicit action for
    the user — out of scope for this skill.

11. **Update this file's "Last synced state" section** with the new monorepo SHA(s), version,
    date, and PR link, as the last step of a successful sync.

## Known gotchas (learned from the 1.1.0 sync, still true unless you find otherwise)

- The monorepo's `tests/sdk/` directory (its own staging ground for SDK tests) had ~37
  pre-existing failures unrelated to any given sync, per its `.workitems/sdk_fixes_needed.md` —
  check current status there before treating that suite as a blocking gate; it's this repo's own
  `tests/unit`/`tests/integration`/`tests/e2e` that actually gate a release.
- CI here (`.github/workflows/test.yml`) blocks on `black`/`isort`/`flake8`; `pylint` runs with
  `--exit-zero` and `bandit` with `|| true` — both non-blocking, don't spend time chasing their
  findings during a sync.
- `docs.yml`'s `markdownlint` step is also non-blocking (`|| true`); its `mkdocs build --strict`
  step is not.

## Delegating to a subagent

A subagent starts with no memory of this conversation. Paste this file's "Last synced state"
section, the classify-noise-vs-functional heuristics from step 2, and the gotchas list directly
into its prompt — don't assume it will load this skill on its own.
