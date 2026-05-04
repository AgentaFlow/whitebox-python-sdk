# Changelog

All notable changes to the WhiteBoxAI Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
