# whitebox-python-sdk
Python language SDK for the AgentaFlow WhiteBox AI platform

```
whiteboxai-python-sdk/
├── src/
│   └── whiteboxai/
│       ├── __init__.py
│       ├── __version__.py
│       ├── client.py
│       ├── monitor.py
│       ├── decorators.py
│       ├── privacy.py
│       ├── offline.py
│       ├── integrations/
│       │   ├── sklearn.py
│       │   ├── pytorch.py
│       │   ├── tensorflow.py
│       │   ├── transformers.py
│       │   ├── langchain.py
│       │   └── boosting.py
│       └── models/          # Pydantic models
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── examples/
│   ├── basic_monitoring.py
│   ├── sklearn_integration.py
│   ├── pytorch_integration.py
│   ├── offline_mode_example.py
│   └── ...
├── docs/
│   ├── getting-started.md
│   ├── integrations.md
│   ├── offline-mode.md
│   └── api-reference.md
├── pyproject.toml
├── setup.py
├── README.md
├── CHANGELOG.md
├── LICENSE
└── .github/
    └── workflows/
        ├── test.yml
        ├── publish.yml
        └── docs.yml
```
