# Changelog

All notable changes to the WhiteBoxXAI Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-07-14

First stable, production release. This release realigns the SDK with the
`whiteboxxai.com` product branding and folds in the fixes/features
developed in the `whitebox-xai-azure` monorepo's `sdk/` directory.

### Changed
- **BREAKING**: Rebranded from `whiteboxai`/`whiteboxai-sdk` (single "x",
  `whiteboxai.io`/`whitebox.agentaflow.com`) to `whiteboxxai`/`whitebox-xai-sdk`
  (double "x", `whiteboxxai.com`). Update imports: `from whiteboxai import ...`
  → `from whiteboxxai import ...`; update the install command to
  `pip install whitebox-xai-sdk`; update the `EXPLAINAI_*` env vars to
  `WHITEBOXXAI_*` (`WHITEBOXXAI_API_KEY`, `WHITEBOXXAI_BASE_URL`).
- **BREAKING**: Switched from a `src/whiteboxai/` layout to a flat
  `whiteboxxai/` layout at the repository root, matching the canonical
  source in `whitebox-xai-azure/sdk/`. Removed the legacy `setup.py`;
  `pyproject.toml` is now the single build configuration.
- Version is now read from installed package metadata
  (`importlib.metadata.version("whitebox-xai-sdk")`) instead of being
  hardcoded in three separate places.
- `Development Status` classifier bumped from Beta to Production/Stable.
- `ModelMonitor.log_prediction()`/`alog_prediction()`/`log_batch()` now call
  the predictions API with the correct `input_data`/`output_data` field
  names (previously sent `inputs`/`outputs`, which the real API doesn't
  accept).

### Added
- `ModelMonitor` local buffering (`buffer_size=`) with `flush()`/`aflush()`
  and context-manager support (`with ModelMonitor(...) as monitor:`
  flushes on exit).
- `ModelMonitor.get_prediction_count()`, `get_drift_reports()`,
  `create_alert_rule()`, and `get_active_alerts()` convenience methods.
- `log_prediction(explain=True)` now actually triggers explanation
  generation via the explanations resource, instead of silently
  accepting and dropping the flag.
- SDK exception classes (`APIError`, `AuthenticationError`, `RateLimitError`,
  `ValidationError`, `NotFoundError`) now carry structured metadata
  (`status_code`, `response`, `request_id`, `retry_after`, `fields`, etc.)
  instead of being bare, attribute-less `Exception` subclasses.
- `Config` now validates `timeout`/`max_retries` are positive.
- MCP (Model Context Protocol) usage documented in `README.md` for
  non-Python integrations, pointing at the companion `whiteboxxai-mcp`
  package.
- `LICENSE` file (MIT), matching `pyproject.toml`'s declared license.

### Fixed
- Fixed an import-time crash: `whiteboxxai.integrations` (and therefore the
  whole SDK) failed to import for anyone without PyTorch or TensorFlow
  installed, due to bad optional-dependency fallback stubs in
  `integrations/pytorch.py` and `integrations/tensorflow.py`, and an
  unguarded/unbound name in `integrations/langchain.py`.
- `README.md` code samples fixed throughout: wrong import path
  (`whiteboxai` → `whiteboxxai`), wrong class name (`WhiteBoxAI` →
  `WhiteBoxXAI`), and wrong install command.
- Corrected `WhiteBoxXAI_*`-cased env var references (wrong case, and in
  one place the wrong variable name entirely) to the real
  `WHITEBOXXAI_API_KEY`/`WHITEBOXXAI_BASE_URL` across guides and examples.
- `docs/` guides (`api-reference.md`, `getting-started.md`, `index.md`,
  `integrations.md`, `offline-mode.md`) rebranded from the stale
  `whiteboxai.io`/`agentaflow.com`/`EXPLAINAI_*` naming.

## [0.2.1] - 2026-05-03

No changelog entry was recorded for this release at the time. Reconstructed
from the git history between the `0.2.0` and `0.2.1` tags:

### Added
- `AgentWorkflows` resource and client binding for multi-agent workflow
  tracking.

### Fixed
- e2e pytest exit-code handling.
- Import normalization and formatting cleanup across integrations and
  examples.

## [0.2.0] - 2026-02-10

### Added
- **Git Integration**: Automatic Git context detection for model versioning
  - `GitContext` class for repository metadata
  - `detect_git_context()` function for auto-detection
  - `validate_git_context()` for validation
  - Support for both GitPython and subprocess fallback
- **CrewAI Multi-Agent Monitoring**: Full support for CrewAI workflows
  - `CrewAIMonitor` class for monitoring crews
  - Automatic agent and task tracking
  - Agent-to-agent interaction logging
  - Token usage and cost tracking
  - Workflow analytics
- **LangChain Multi-Agent Support**: Enhanced LangChain integration
  - `MultiAgentCallbackHandler` for agent execution tracking
  - `LangGraphMultiAgentMonitor` for LangGraph workflows
  - Agent-to-agent handoff monitoring
  - Tool call tracking
  - `monitor_langchain_agent()` helper function
- **Documentation**: Complete MkDocs setup with Material theme
  - Comprehensive navigation structure
  - API reference integration
  - Code highlighting and copy buttons
  - Dark/light mode support

### Changed
- **BREAKING**: Fixed import paths from `explainai.*` to `whiteboxai.*`
  - Users must update imports: `from explainai.client` → `from whiteboxai.client`
- Updated dependencies:
  - httpx: >=0.24.0 → >=0.25.0
  - numpy: >=1.24.0 (aligned with latest stable)
  - Added pandas>=1.3.0 (core dependency)
  - Added tenacity>=8.0.0 (core dependency)
- Enhanced optional dependencies:
  - Added git extra: `pip install whiteboxai-sdk[git]`
  - Added crewai extra: `pip install whiteboxai-sdk[crewai]`
  - Updated all extra: includes git, crewai, and all integrations

### Fixed
- Import errors due to incorrect package naming (explainai vs whiteboxai)
- Missing MkDocs configuration causing documentation build failures
- Incomplete integration exports in `whiteboxai.integrations`

### Documentation
- Created comprehensive MkDocs configuration
- Added index page with quick start examples
- Organized documentation with clear navigation
- Added examples for Git integration, CrewAI, and LangChain agents

## [0.1.0] - 2026-01-05

### Added
- Initial release of WhiteBoxAI Python SDK
- Basic monitoring functionality
- Scikit-learn integration
- PyTorch integration
- TensorFlow/Keras integration
- Hugging Face Transformers integration
- LangChain integration
- XGBoost/LightGBM integration
- Offline mode support
- Privacy filters
- Caching support
- Example code
- Documentation

### Security
- PII detection and masking
- Secure API key handling

[Unreleased]: https://github.com/AgentaFlow/whitebox-python-sdk/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/AgentaFlow/whitebox-python-sdk/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AgentaFlow/whitebox-python-sdk/releases/tag/v0.1.0
