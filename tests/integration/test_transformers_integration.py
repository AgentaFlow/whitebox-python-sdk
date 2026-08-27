"""
Unit tests for the Hugging Face Transformers integration's import guard.

Focused on the exception-handling fix for a tokenizers/transformers version mismatch: transformers'
top-level API is a lazy module whose attribute access re-raises such mismatches as RuntimeError,
not ImportError. A guard that only catches ImportError lets that RuntimeError escape and abort
every integration registered after this one in integrations/__init__.py (crewai, boosting,
langchain_agents) — see sdk/whiteboxxai/integrations/transformers.py.
"""

import importlib
import sys

import whiteboxxai.integrations.transformers as transformers_integration


def test_transformers_available_flag_exists():
    """The module always defines TRANSFORMERS_AVAILABLE, whether or not the
    real transformers package is installed in this environment."""
    assert hasattr(transformers_integration, "TRANSFORMERS_AVAILABLE")
    assert isinstance(transformers_integration.TRANSFORMERS_AVAILABLE, bool)


def test_runtime_error_during_import_is_caught():
    """Simulate transformers' lazy-module loader raising RuntimeError (its
    real behavior on a tokenizers/transformers ABI mismatch) and confirm the
    integration module absorbs it instead of letting it escape."""

    class ExplodingTransformersModule:
        def __getattr__(self, name):
            raise RuntimeError("simulated tokenizers/transformers ABI mismatch")

    original = sys.modules.get("transformers")
    sys.modules["transformers"] = ExplodingTransformersModule()
    try:
        importlib.reload(transformers_integration)
        assert transformers_integration.TRANSFORMERS_AVAILABLE is False
        assert transformers_integration.Pipeline is object
        assert transformers_integration.PreTrainedModel is object
        assert transformers_integration.PreTrainedTokenizer is object
    finally:
        if original is not None:
            sys.modules["transformers"] = original
        else:
            sys.modules.pop("transformers", None)
        importlib.reload(transformers_integration)
