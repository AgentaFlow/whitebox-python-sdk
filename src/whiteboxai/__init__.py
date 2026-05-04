"""
WhiteBoxAI Python SDK

Official Python SDK for WhiteBoxAI - AI Observability & Explainability Platform.
"""

__version__ = "0.2.0"
__author__ = "WhiteBoxAI Team"
__license__ = "MIT"

from whiteboxai.client import WhiteBoxAI
from whiteboxai.decorators import monitor_model, monitor_prediction
from whiteboxai.git_utils import GitContext, detect_git_context, validate_git_context
from whiteboxai.monitor import ModelMonitor

__all__ = [
    "WhiteBoxAI",
    "ModelMonitor",
    "monitor_model",
    "monitor_prediction",
    "GitContext",
    "detect_git_context",
    "validate_git_context",
]
