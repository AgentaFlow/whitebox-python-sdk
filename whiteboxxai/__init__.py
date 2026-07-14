"""
WhiteBoxXAI Python SDK

Official Python SDK for WhiteBoxXAI - AI Observability & Explainability Platform.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("whitebox-xai-sdk")
except PackageNotFoundError:
    # Package isn't installed (e.g. running directly from a source checkout).
    __version__ = "0.0.0.dev0"

__author__ = "WhiteBoxXAI Team"
__license__ = "MIT"

from whiteboxxai.client import WhiteBoxXAI
from whiteboxxai.decorators import monitor_model, monitor_prediction
from whiteboxxai.git_utils import GitContext, detect_git_context, validate_git_context
from whiteboxxai.monitor import ModelMonitor

__all__ = [
    "WhiteBoxXAI",
    "ModelMonitor",
    "monitor_model",
    "monitor_prediction",
    "GitContext",
    "detect_git_context",
    "validate_git_context",
]
