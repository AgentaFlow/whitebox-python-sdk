# WhiteBoxXAI SDK - Separate Repository Migration Guide

This document provides comprehensive instructions for migrating the WhiteBoxXAI Python SDK to its own repository for easier distribution and better developer experience.

---

## Why Separate the SDK?

### Benefits of Separate SDK Repository

#### 1. **Easier Distribution & Installation**
```bash
# Users can install directly from PyPI
pip install whiteboxai

# Or from GitHub
pip install git+https://github.com/AgentaFlow/whiteboxai-python-sdk.git
```

#### 2. **Independent Versioning**
- SDK can have its own semantic versioning (v1.2.3)
- Backend API changes don't force SDK updates
- Users can pin to stable SDK versions while you iterate on backend
- Clear separation of concerns between platform and client library

#### 3. **Cleaner Developer Experience**
- SDK contributors don't need to clone entire platform
- Smaller repository = faster clones
- Separate issue tracking for SDK vs platform
- Independent CI/CD pipelines
- Focused documentation and examples

#### 4. **Better Package Management**
- Cleaner package structure for PyPI
- Simplified dependency management
- Better testing isolation
- Industry-standard approach

#### 5. **PyPI Publishing**
Separate repo makes PyPI publishing cleaner:
```bash
# Build and publish
python -m build
twine upload dist/*
```

---

## Recommended Repository Structure

### Main Platform Repository (Current)
```
whitebox-local-demo/
├── backend/              # FastAPI application
├── frontend/             # React/Node.js UI
├── docker-compose.yml    # Platform deployment
├── README.md             # Platform overview
├── docs/                 # Platform documentation
├── tests/                # Platform tests
└── infrastructure/       # Infrastructure as Code (Bicep, Azure)
```

### SDK Repository (NEW)
```
whiteboxai-python-sdk/
├── src/
│   └── whiteboxai/
│       ├── __init__.py
│       ├── __version__.py
│       ├── client.py              # WhiteBoxAI client
│       ├── monitor.py             # ModelMonitor
│       ├── config.py              # Configuration
│       ├── decorators.py          # @monitor_model, @monitor_prediction
│       ├── privacy.py             # PIIDetector, DataMasker
│       ├── offline.py             # OfflineQueue, OfflineManager
│       ├── integrations/
│       │   ├── __init__.py
│       │   ├── sklearn.py         # Scikit-learn integration
│       │   ├── pytorch.py         # PyTorch integration
│       │   ├── tensorflow.py      # TensorFlow/Keras integration
│       │   ├── transformers.py    # Hugging Face integration
│       │   ├── langchain.py       # LangChain integration
│       │   └── boosting.py        # XGBoost/LightGBM integration
│       ├── models/                # Pydantic models
│       │   ├── __init__.py
│       │   ├── prediction.py
│       │   ├── model.py
│       │   └── response.py
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           └── validation.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_client.py
│   │   ├── test_monitor.py
│   │   ├── test_privacy.py
│   │   └── test_offline.py
│   ├── integration/
│   │   ├── test_sklearn.py
│   │   ├── test_pytorch.py
│   │   ├── test_tensorflow.py
│   │   └── test_api.py
│   └── e2e/
│       └── test_workflow.py
├── examples/
│   ├── README.md
│   ├── basic_monitoring.py
│   ├── sklearn_integration.py
│   ├── pytorch_integration.py
│   ├── tensorflow_integration.py
│   ├── transformers_integration.py
│   ├── langchain_integration.py
│   ├── offline_mode_example.py
│   ├── privacy_example.py
│   └── notebooks/
│       ├── getting_started.ipynb
│       ├── advanced_monitoring.ipynb
│       └── llm_monitoring.ipynb
├── docs/
│   ├── index.md
│   ├── getting-started.md
│   ├── installation.md
│   ├── api-reference.md
│   ├── integrations/
│   │   ├── sklearn.md
│   │   ├── pytorch.md
│   │   ├── tensorflow.md
│   │   ├── transformers.md
│   │   ├── langchain.md
│   │   └── boosting.md
│   ├── features/
│   │   ├── offline-mode.md
│   │   ├── privacy-filters.md
│   │   ├── decorators.md
│   │   └── async-support.md
│   ├── guides/
│   │   ├── production-deployment.md
│   │   ├── best-practices.md
│   │   └── troubleshooting.md
│   └── changelog.md
├── .github/
│   └── workflows/
│       ├── test.yml              # Run tests on PR
│       ├── publish.yml           # Publish to PyPI on release
│       ├── docs.yml              # Build and deploy docs
│       └── lint.yml              # Code quality checks
├── pyproject.toml                # Modern Python packaging
├── setup.py                      # Legacy support
├── README.md                     # SDK-focused README
├── CHANGELOG.md                  # Version history
├── LICENSE                       # MIT License
├── MANIFEST.in                   # Package includes
├── .gitignore
├── .pre-commit-config.yaml       # Pre-commit hooks
└── mkdocs.yml                    # Documentation config
```

---

## Migration Strategy

### Phase 1: Create New Repository

#### Step 1.1: Create Repository on GitHub
```bash
# On GitHub, create new repository:
# Name: whiteboxxai-python-sdk
# Description: Official Python SDK for WhiteBoxXAI - AI Observability & Explainability Platform
# Visibility: Public
# Initialize with: README, .gitignore (Python), License (MIT)
```

#### Step 1.2: Clone and Setup Local Repository
```bash
# Clone the new repository
git clone https://github.com/AgentaFlow/whiteboxxai-python-sdk.git
cd whiteboxxai-python-sdk

# Create branch for migration
git checkout -b feature/initial-migration
```

#### Step 1.3: Copy SDK Files
```bash
# Copy SDK source code
cp -r ../whitebox-local-demo/sdk/whiteboxxai ./src/

# Copy examples
cp -r ../whitebox-local-demo/examples/sdk_examples ./examples/

# Copy tests
cp -r ../whitebox-local-demo/tests/sdk ./tests/

# Copy documentation
cp ../whitebox-local-demo/sdk/README.md ./README.md
cp -r ../whitebox-local-demo/docs/SDK_*.md ./docs/
```

#### Step 1.4: Restructure to Modern Python Layout
```bash
# Create src layout (recommended by PyPA)
mkdir -p src/whiteboxxai
mv whiteboxxai/* src/whiteboxxai/

# Create version file
cat > src/whiteboxxai/__version__.py << 'EOF'
"""WhiteBoxXAI SDK version information."""

__version__ = "0.1.0"
__author__ = "AgentaFlow"
__email__ = "support@whiteboxxai.com"
__license__ = "MIT"
EOF

# Update __init__.py to include version
cat >> src/whiteboxxai/__init__.py << 'EOF'

from .__version__ import __version__, __author__, __email__

__all__ = [
    "WhiteBoxXAI",
    "ModelMonitor",
    "__version__",
]
EOF
```

#### Step 1.5: Create Modern pyproject.toml
```bash
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "whiteboxxai"
version = "0.1.0"
description = "Official Python SDK for WhiteBoxXAI - AI Observability & Explainability Platform"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "AgentaFlow", email = "support@whiteboxxai.com"}
]
keywords = [
    "ai",
    "ml",
    "machine-learning",
    "observability",
    "explainability",
    "xai",
    "monitoring",
    "mlops",
    "model-monitoring",
    "drift-detection",
    "bias-detection"
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

dependencies = [
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "numpy>=1.20.0",
    "tenacity>=8.0.0",
]

[project.optional-dependencies]
sklearn = [
    "scikit-learn>=1.0.0",
    "shap>=0.41.0",
]
pytorch = [
    "torch>=1.9.0",
]
tensorflow = [
    "tensorflow>=2.6.0",
]
transformers = [
    "transformers>=4.20.0",
    "torch>=1.9.0",
]
langchain = [
    "langchain>=0.1.0",
]
boosting = [
    "xgboost>=1.5.0",
    "lightgbm>=3.3.0",
]
all = [
    "whiteboxxai[sklearn,pytorch,tensorflow,transformers,langchain,boosting]",
]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
    "build>=0.10.0",
    "twine>=4.0.0",
]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0",
    "mkdocstrings[python]>=0.24.0",
]

[project.urls]
Homepage = "https://whiteboxxai.com"
Documentation = "https://docs.whiteboxxai.com/sdk"
Repository = "https://github.com/AgentaFlow/whiteboxxai-python-sdk"
Issues = "https://github.com/AgentaFlow/whiteboxxai-python-sdk/issues"
Changelog = "https://github.com/AgentaFlow/whiteboxxai-python-sdk/blob/main/CHANGELOG.md"

[tool.setuptools]
packages = ["whiteboxxai"]
package-dir = {"" = "src"}

[tool.setuptools.package-data]
whiteboxxai = ["py.typed"]

[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']
include = '\.pyi?$'

[tool.ruff]
line-length = 100
target-version = "py39"
select = ["E", "F", "I", "N", "W", "UP"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=whiteboxxai --cov-report=term-missing --cov-report=html"

[tool.coverage.run]
source = ["src/whiteboxxai"]
omit = ["*/tests/*", "*/test_*.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
EOF
```

#### Step 1.6: Create setup.py (for backward compatibility)
```bash
cat > setup.py << 'EOF'
"""
WhiteBoxXAI Python SDK Setup
For modern builds, use: python -m build
"""
from setuptools import setup

# All configuration is in pyproject.toml
# This file exists for backward compatibility
setup()
EOF
```

#### Step 1.7: Create MANIFEST.in
```bash
cat > MANIFEST.in << 'EOF'
include README.md
include LICENSE
include CHANGELOG.md
include pyproject.toml
include setup.py
recursive-include src/whiteboxxai *.py
recursive-include src/whiteboxxai py.typed
recursive-include tests *.py
recursive-include examples *.py *.ipynb
recursive-include docs *.md
prune tests/__pycache__
prune src/whiteboxxai/__pycache__
EOF
```

#### Step 1.8: Create CHANGELOG.md
```bash
cat > CHANGELOG.md << 'EOF'
# Changelog

All notable changes to the WhiteBoxXAI Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-05

### Added
- Initial release of WhiteBoxXAI Python SDK
- Core client and model monitoring functionality
- Framework integrations:
  - Scikit-learn
  - PyTorch
  - TensorFlow/Keras
  - Hugging Face Transformers
  - LangChain
  - XGBoost/LightGBM
- Privacy features (PII detection and masking)
- Offline mode with SQLite queue
- Decorator-based monitoring
- Async/sync interfaces
- Comprehensive examples and documentation

[0.1.0]: https://github.com/AgentaFlow/whiteboxxai-python-sdk/releases/tag/v0.1.0
EOF
```

---

### Phase 2: Setup CI/CD

#### Step 2.1: Create GitHub Actions for Testing
```bash
mkdir -p .github/workflows

cat > .github/workflows/test.yml << 'EOF'
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.9', '3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run tests
        run: |
          pytest --cov=whiteboxxai --cov-report=xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
EOF
```

#### Step 2.2: Create GitHub Actions for Publishing
```bash
cat > .github/workflows/publish.yml << 'EOF'
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build-and-publish:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Check package
        run: twine check dist/*

      - name: Publish to Test PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.TEST_PYPI_API_TOKEN }}
        run: twine upload --repository testpypi dist/*

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
EOF
```

#### Step 2.3: Create GitHub Actions for Linting
```bash
cat > .github/workflows/lint.yml << 'EOF'
name: Lint

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  lint:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install black ruff mypy

      - name: Run Black
        run: black --check src/ tests/

      - name: Run Ruff
        run: ruff check src/ tests/

      - name: Run MyPy
        run: mypy src/whiteboxxai
EOF
```

#### Step 2.4: Create GitHub Actions for Documentation
```bash
cat > .github/workflows/docs.yml << 'EOF'
name: Build Docs

on:
  push:
    branches: [ main ]

jobs:
  build-docs:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[docs]"

      - name: Build documentation
        run: mkdocs build

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
EOF
```

---

### Phase 3: Update Main Platform Repository

#### Step 3.1: Update Main README
```markdown
# WhiteBoxXAI Platform

AI Observability & Explainability Platform for production ML models.

## Components

- **Platform**: FastAPI backend + React frontend (this repository)
- **Python SDK**: [whiteboxxai-python-sdk](https://github.com/AgentaFlow/whiteboxxai-python-sdk)
- **Documentation**: [docs.whiteboxxai.com](https://docs.whiteboxxai.com)

## Quick Start

### 1. Install SDK
```bash
pip install whiteboxxai
```

### 2. Monitor Your Models
```python
from whiteboxxai import WhiteBoxXAI, ModelMonitor

client = WhiteBoxXAI(api_key="your-api-key")
monitor = ModelMonitor(client)

# Register model
model_id = monitor.register_model(
    name="fraud_detection",
    model_type="classification"
)

# Log predictions
monitor.log_prediction(
    inputs={"amount": 100.0},
    output={"fraud": False}
)
```

### 3. View in Dashboard
Visit http://localhost:3000 to see real-time monitoring, drift detection, and explainability.

## For Developers

See [DEVELOPMENT.md](DEVELOPMENT.md) for platform development setup.

## Documentation

- **SDK Documentation**: https://github.com/AgentaFlow/whiteboxxai-python-sdk
- **Platform Documentation**: [docs/](docs/)
- **API Reference**: [API_REFERENCE.md](docs/API_REFERENCE.md)

## License

MIT License - see [LICENSE](LICENSE)
```

#### Step 3.2: Remove SDK Directory from Main Repo
```bash
cd whitebox-local-demo

# Create branch
git checkout -b feature/remove-sdk-directory

# Remove SDK directory
git rm -r sdk/

# Update .gitignore if needed
echo "" >> .gitignore
echo "# SDK is now in separate repository" >> .gitignore
echo "sdk/" >> .gitignore

# Commit changes
git add .
git commit -m "Move SDK to separate repository

SDK has been moved to https://github.com/AgentaFlow/whiteboxxai-python-sdk
for better distribution and independent versioning.

Users should now install via: pip install whiteboxxai"

# Push and create PR
git push origin feature/remove-sdk-directory
```

#### Step 3.3: Update docker-compose.yml (if SDK is used in tests)
```yaml
# docker-compose.yml
services:
  backend:
    # ... existing config ...
    volumes:
      - ./backend:/app/backend
      # Remove SDK volume mount if it exists
    environment:
      # Add if testing against local SDK
      - INSTALL_LOCAL_SDK=false
```

#### Step 3.4: Update Documentation Links
Update all references to SDK in the main repository:
- `README.md` → Point to SDK repository
- `docs/GETTING_STARTED.md` → Update SDK installation
- `docs/SDK_DOCUMENTATION.md` → Add deprecation notice pointing to SDK repo
- `examples/` → Update import statements and installation instructions

---

### Phase 4: Publishing to PyPI

#### Step 4.1: Register on PyPI
```bash
# Register accounts (if not already done)
# 1. Go to https://pypi.org/account/register/
# 2. Go to https://test.pypi.org/account/register/
# 3. Set up 2FA for security

# Generate API tokens
# PyPI → Account Settings → API tokens → Add API token
# Scope: Entire account or specific project
# Copy token and save securely
```

#### Step 4.2: Test Build Locally
```bash
cd whiteboxxai-python-sdk

# Install build tools
pip install build twine

# Build package
python -m build

# Check build
twine check dist/*

# Should output:
# Checking dist/whiteboxxai-0.1.0-py3-none-any.whl: PASSED
# Checking dist/whiteboxxai-0.1.0.tar.gz: PASSED
```

#### Step 4.3: Publish to Test PyPI
```bash
# Upload to Test PyPI first
twine upload --repository testpypi dist/*

# Enter username: __token__
# Enter password: <your Test PyPI API token>

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ whiteboxxai

# Test that it works
python -c "from whiteboxxai import WhiteBoxXAI; print('Success!')"
```

#### Step 4.4: Publish to Production PyPI
```bash
# Upload to production PyPI
twine upload dist/*

# Enter username: __token__
# Enter password: <your PyPI API token>

# Test installation
pip install whiteboxxai

# Verify
python -c "from whiteboxxai import WhiteBoxXAI, __version__; print(__version__)"
```

#### Step 4.5: Setup GitHub Secrets for Automated Publishing
```bash
# On GitHub repository settings:
# Settings → Secrets and variables → Actions → New repository secret

# Add secrets:
# - TEST_PYPI_API_TOKEN: <your Test PyPI token>
# - PYPI_API_TOKEN: <your PyPI token>

# Now releases will automatically publish to PyPI
```

---

### Phase 5: Documentation Setup

#### Step 5.1: Create MkDocs Configuration
```bash
cat > mkdocs.yml << 'EOF'
site_name: WhiteBoxXAI Python SDK
site_description: Official Python SDK for WhiteBoxXAI - AI Observability & Explainability
site_author: AgentaFlow
site_url: https://docs.whiteboxxai.com/sdk

repo_name: AgentaFlow/whiteboxxai-python-sdk
repo_url: https://github.com/AgentaFlow/whiteboxxai-python-sdk

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - search.suggest
    - search.highlight
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
            show_root_heading: true

nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Quick Start: getting-started/quickstart.md
    - Configuration: getting-started/configuration.md
  - Integrations:
    - Scikit-learn: integrations/sklearn.md
    - PyTorch: integrations/pytorch.md
    - TensorFlow: integrations/tensorflow.md
    - Hugging Face: integrations/transformers.md
    - LangChain: integrations/langchain.md
    - XGBoost/LightGBM: integrations/boosting.md
  - Features:
    - Offline Mode: features/offline-mode.md
    - Privacy Filters: features/privacy.md
    - Decorators: features/decorators.md
    - Async Support: features/async.md
  - Guides:
    - Production Deployment: guides/production.md
    - Best Practices: guides/best-practices.md
    - Troubleshooting: guides/troubleshooting.md
  - API Reference:
    - Client: api/client.md
    - Monitor: api/monitor.md
    - Integrations: api/integrations.md
  - Changelog: changelog.md

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - admonition
  - pymdownx.details
  - pymdownx.tabbed:
      alternate_style: true
  - attr_list
  - md_in_html
EOF
```

#### Step 5.2: Create Documentation Index
```bash
mkdir -p docs/getting-started docs/integrations docs/features docs/guides docs/api

cat > docs/index.md << 'EOF'
# WhiteBoxXAI Python SDK

Official Python SDK for integrating WhiteBoxXAI monitoring into your ML applications.

## Features

- 🚀 **Easy Integration** - Monitor models with just a few lines of code
- 📊 **Framework Support** - Native integrations for Scikit-learn, PyTorch, TensorFlow, XGBoost, and more
- 🎯 **Decorator-based Monitoring** - Zero-code-change monitoring with decorators
- ⚡ **Async/Sync Interfaces** - Support for both synchronous and asynchronous workflows
- 🔒 **Privacy-First** - Built-in PII detection and data masking
- 💾 **Offline Mode** - Queue predictions when API is unavailable
- 📈 **Drift Detection** - Automatic model and data drift monitoring

## Quick Start

```python
from whiteboxxai import WhiteBoxXAI, ModelMonitor

# Initialize client
client = WhiteBoxXAI(api_key="your-api-key")

# Create monitor
monitor = ModelMonitor(client)

# Register model
model_id = monitor.register_model(
    name="fraud_detection",
    model_type="classification",
    framework="sklearn"
)

# Log predictions
monitor.log_prediction(
    inputs={"amount": 100.0, "merchant": "store_123"},
    output={"fraud_probability": 0.15, "prediction": "legitimate"}
)
```

## Installation

```bash
pip install whiteboxxai

# With specific framework support
pip install whiteboxxai[sklearn]
pip install whiteboxxai[pytorch]
pip install whiteboxxai[all]  # All integrations
```

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [Framework Integrations](integrations/sklearn.md)
- [API Reference](api/client.md)
EOF
```

---

### Phase 6: Pre-commit Hooks and Code Quality

#### Step 6.1: Create .pre-commit-config.yaml
```bash
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2.0.0]
EOF

# Install pre-commit
pip install pre-commit
pre-commit install
```

---

## Version Release Process

### Creating a New Release

#### 1. Update Version Number
```bash
# Update version in src/whiteboxxai/__version__.py
# Update version in pyproject.toml
# Update CHANGELOG.md with new version
```

#### 2. Commit Changes
```bash
git add .
git commit -m "chore: bump version to 0.2.0"
git push origin main
```

#### 3. Create Git Tag
```bash
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

#### 4. Create GitHub Release
```bash
# On GitHub:
# Releases → Draft a new release
# Tag: v0.2.0
# Title: WhiteBoxXAI Python SDK v0.2.0
# Description: Copy from CHANGELOG.md
# Publish release

# This will automatically trigger the publish workflow
# and upload to PyPI
```

---

## Maintenance and Development

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/AgentaFlow/whiteboxxai-python-sdk.git
cd whiteboxxai-python-sdk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
black src/ tests/
ruff check src/ tests/
mypy src/whiteboxxai

# Build documentation locally
mkdocs serve
# Visit http://localhost:8000
```

### Testing Against Local Platform

```bash
# Terminal 1: Run platform
cd whitebox-local-demo
docker-compose up

# Terminal 2: Test SDK
cd whiteboxxai-python-sdk
export WHITEBOXXAI_BASE_URL=http://localhost:8000
pytest tests/integration/
```

---

## Best Practices

### 1. Semantic Versioning
- **MAJOR** (1.0.0): Breaking API changes
- **MINOR** (0.1.0): New features, backward compatible
- **PATCH** (0.0.1): Bug fixes, backward compatible

### 2. Changelog Management
- Update CHANGELOG.md for every release
- Use categories: Added, Changed, Deprecated, Removed, Fixed, Security
- Include migration guides for breaking changes

### 3. Backward Compatibility
- Deprecate before removing features
- Use warnings for deprecated functionality
- Maintain compatibility for at least 2 minor versions

### 4. Documentation
- Update docs with every feature addition
- Include code examples in docstrings
- Keep README.md concise, detailed docs in MkDocs

### 5. Testing
- Maintain >80% code coverage
- Test all integrations separately
- Include integration tests with mock API
- Add end-to-end tests for critical workflows

---

## Support and Community

### Documentation
- **SDK Docs**: https://docs.whiteboxxai.com/sdk
- **Platform Docs**: https://docs.whiteboxxai.com
- **API Reference**: https://api.whiteboxxai.com/docs

### Community
- **GitHub Issues**: https://github.com/AgentaFlow/whiteboxxai-python-sdk/issues
- **Discussions**: https://github.com/AgentaFlow/whiteboxxai-python-sdk/discussions
- **Discord**: https://discord.gg/whiteboxxai (if available)

### Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

## Success Metrics

Track these metrics after migration:

### Distribution Metrics
- PyPI downloads per month
- GitHub stars and forks
- Issue resolution time
- PR merge time

### Quality Metrics
- Test coverage percentage
- Number of open issues
- Number of open PRs
- Code quality scores

### Community Metrics
- Number of contributors
- Community engagement (issues, PRs, discussions)
- Documentation page views
- SDK adoption rate

---

## Comparison: Before vs After

| Aspect | Before (Monorepo) | After (Separate Repo) |
|--------|-------------------|----------------------|
| Installation | Clone entire platform | `pip install whiteboxxai` |
| Updates | Coupled with platform | Independent versioning |
| Contributing | Need platform knowledge | SDK-focused contributions |
| CI/CD | Shared with platform | Dedicated pipelines |
| Documentation | Mixed with platform | SDK-specific docs |
| Testing | Platform tests | Isolated SDK tests |
| Release Cycle | Tied to platform | Independent releases |
| Repository Size | Large (~500MB+) | Small (~5MB) |
| Clone Time | Minutes | Seconds |
| Focus | Divided | SDK-specific |

---

## Industry Examples

### Similar Approaches by Leading Companies

**Stripe**:
- Platform: Multiple microservices
- SDKs: `stripe-python`, `stripe-go`, `stripe-ruby`, `stripe-node`
- Each SDK in separate repository

**Sentry**:
- Platform: Sentry error tracking service
- SDKs: `sentry-python`, `sentry-javascript`, `sentry-java`
- Independent versioning and release cycles

**OpenAI**:
- Platform: OpenAI API service
- SDKs: `openai-python`, `openai-node`
- Clean separation of concerns

**AWS**:
- Platform: AWS Cloud Services
- SDKs: Hundreds of separate SDK repositories
- Language-specific repositories

---

## Conclusion

Migrating the WhiteBoxXAI Python SDK to its own repository is a strategic decision that will:

1. ✅ Improve user experience with simpler installation
2. ✅ Enable independent versioning and release cycles
3. ✅ Reduce repository complexity and size
4. ✅ Attract more SDK-focused contributors
5. ✅ Follow industry best practices
6. ✅ Prepare for future multi-language SDK support
7. ✅ Enable professional PyPI distribution
8. ✅ Improve documentation and discoverability

The migration is straightforward and can be completed in a few days with minimal disruption to users.

---

**Last Updated**: January 5, 2026
**Version**: 1.0
**Status**: Ready for Implementation
