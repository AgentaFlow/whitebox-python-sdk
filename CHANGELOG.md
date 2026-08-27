# Changelog

All notable changes to the WhiteBoxXAI Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2026-08-27

Ports the MCP server M2 domain-coverage work (issue #141), the ISO 42001
governance alignment, the Fable security audit's PII-detection hardening,
and the async-explanation-generation feature (issue #137) developed in the
`whitebox-xai-azure` monorepo's `sdk/` directory since the 1.1.0 cut.

### Fixed
- **Offline mode was completely non-functional**: `OfflineManager.sync()`
  called `client._api_predict()`/`_api_log_batch()`/`_api_register_model()`/
  `_api_update_baseline()`, none of which existed on `WhiteBoxXAI`, and
  `ModelsResource.register()`/`update_baseline()`/`PredictionsResource.log()`/
  `log_batch()` never enqueued on a connection failure in the first place
  (there was no `APIConnectionError` exception type to catch, and
  `BaseResource._request_or_queue()` didn't exist). All four write paths now
  enqueue through the offline queue on connection failure, and
  `OfflineManager.sync()` can actually replay them.
- `OfflineQueue.dequeue()` used a plain `SELECT` with no locking, so two
  concurrent callers (threads/processes sharing the same queue.db) could
  claim and double-process the same row. Now claims rows atomically via
  `BEGIN IMMEDIATE`, and a claim left in `processing` for more than 30
  minutes (a crashed process) is automatically reclaimed as `pending` the
  next time the queue is opened.
- **PII detection**: the credit-card pattern had no Luhn checksum
  validation (any 16-digit run matched, e.g. a phone/account number), and
  the IPv4 pattern had no octet-range validation (`999.999.999.999`
  matched). The email pattern's TLD character class contained a literal
  `|` (`[A-Z|a-z]{2,}`), fixed to `[A-Za-z]{2,}`.
- `integrations.langchain_agents`'s `MultiAgentCallbackHandler`,
  `LangGraphMultiAgentMonitor`, and `monitor_langchain_agent()` called
  `client.agent_workflows.*`, which didn't exist on `WhiteBoxXAI` — every
  real invocation raised `AttributeError`. Fixed by adding
  `AgentWorkflowsResource` (see Added).
- `integrations.transformers` only caught `ImportError` around the
  `transformers` import; a tokenizers/transformers version mismatch raises
  `RuntimeError` from transformers' lazy-module loader instead, which, left
  uncaught, aborted the rest of `integrations/__init__.py` mid-sequence and
  silently dropped every integration registered after it (crewai, boosting,
  langchain_agents). Now also caught.
- `ModelMonitor.create_alert_rule()` called `AlertsResource.create()` with
  the old request shape (no `severity`, `conditions` as a dict); updated
  for the rewritten `AlertsResource.create()` signature below.

### Added
- `client.risk_register`, `client.governance` (ISO 42001 alignment),
  `client.llm`, `client.rag`, `client.safety`, `client.llm_xai`,
  `client.agent_workflows`, `client.metrics` — 8 new resource classes
  backed by their corresponding `backend/api/v1/*` routers.
- `ExplanationsResource.generate_async()`/`agenerate_async()`: starts
  non-blocking SHAP/LIME computation on a Celery task and returns
  immediately with a `pending` explanation record; poll `get()`/`aget()`
  for the result.
- `integrations.langchain`'s `log_llm_call()`/`log_rag_retrieval()` now
  route to the real `client.llm.log_call()`/`client.rag.log_retrieval()`
  observability endpoints (queryable via `client.llm.get_stats()` and the
  LLM/RAG dashboards) instead of the generic prediction-logging fallback.

### Changed
- **BREAKING**: `AlertsResource.create()` now requires a `severity` argument
  and takes `conditions` as a `List[Dict]` (previously an unstructured
  `Dict`); `create()`/`list()` now target `/api/v1/alerts/rules` instead of
  `/api/v1/alerts` (the old endpoint didn't exist on a live backend).
  `AlertsResource` also gains `get_rule()`, `update_rule()`, `delete_rule()`,
  `evaluate_rule()`, `list_instances()`, `get_instance()`, `acknowledge()`,
  `resolve()`, `snooze()`, and `statistics()`.

## [1.1.0] - 2026-07-30

Ports integration fixes and compatibility improvements developed in the
`whitebox-xai-azure` monorepo's `sdk/` directory since the 1.0.0 cut.

### Fixed
- **XGBoost/LightGBM**: `XGBoostMonitor.predict()`/`LightGBMMonitor.predict()`
  and `wrap_xgboost_model()`/`wrap_lightgbm_model()` called a nonexistent
  `log_predictions()` method (raising `AttributeError`) instead of
  `ModelMonitor.log_batch()` — the primary documented usage of both
  integrations was broken.
- **TensorFlow/Keras**: `KerasMonitor.predict()` called `log_prediction()`/
  `log_batch()` with invalid keyword arguments (`prediction=`, `actual=`,
  `predictions=`, `actuals=`), raising `TypeError` on every call.
  `KerasMonitor.log_epoch()`/`log_checkpoint()` (and the
  `WhiteBoxXAICallback` training callback) called a `log_custom_metric()`
  method that didn't exist, raising `AttributeError`.
  `KerasMonitor.set_baseline()` raised `TypeError` when computing baseline
  predictions, since it called the parent `ModelMonitor.set_baseline()`
  with two positional arguments against a signature that only accepted one
  (see the `ModelMonitor.set_baseline()` fix below).
  `register_saved_model()` raised `ValueError` unless `self.model` was set,
  even though it exists specifically to register a model saved to disk.
  `wrap_keras_model()` also called `log_batch()` with invalid keyword
  arguments.
- **Hugging Face Transformers**: `TransformersMonitor.log_prediction_transformers()`/
  `log_batch_transformers()` called `log_prediction()`/`log_batch()` with
  invalid keyword arguments, raising `TypeError` on every prediction.
  `TransformersMonitor.set_baseline()` had the same `ModelMonitor.set_baseline()`
  argument-count crash as `KerasMonitor` above.
  `wrap_transformers_pipeline()` reassigned `pipeline.__call__` on a pipeline
  *instance*, which Python never actually invokes (dunder-method lookup
  happens on the type, not the instance) — wrapped pipelines silently never
  logged any predictions.
- **LangChain**: `LangChainMonitor.log_chain_execution()`,
  `log_agent_execution()`, `log_llm_call()`, `log_tool_call()`, and
  `log_rag_retrieval()` all passed `prediction=` to `log_prediction()`,
  which only accepts `output=` — all five convenience-logging methods
  raised `TypeError`.
- `OfflineQueue.clear_completed()` used an exclusive day-boundary comparison
  (`created_at < ...`), so a completed record exactly `older_than_days` old
  was never cleared. Now inclusive (`<=`).

### Added
- `whiteboxxai.integrations.langchain`/`langchain_agents` now try
  `langchain_core.*` first (the modern, lighter-weight package) and fall
  back to the legacy `langchain.callbacks.base`/`langchain.schema.*`
  locations, so the SDK works with `langchain-core`-only installs.
  `AgentExecutor`/`Chain` (higher-level `langchain`-only abstractions) are
  probed independently so their absence no longer disables the rest of the
  integration.
- `ModelMonitor.set_baseline()` accepts an optional `labels` argument for
  baseline predictions/labels.
- `KerasMonitor.log_custom_metric()`.

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

[Unreleased]: https://github.com/AgentaFlow/whitebox-python-sdk/compare/1.1.0...HEAD
[1.1.0]: https://github.com/AgentaFlow/whitebox-python-sdk/compare/1.0.0...1.1.0
[1.0.0]: https://github.com/AgentaFlow/whitebox-python-sdk/compare/0.2.1...1.0.0
[0.2.1]: https://github.com/AgentaFlow/whitebox-python-sdk/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/AgentaFlow/whitebox-python-sdk/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/AgentaFlow/whitebox-python-sdk/releases/tag/0.1.0
