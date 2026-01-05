"""
WhiteBoxAI Python SDK

Official Python SDK for WhiteBoxAI - AI Observability & Explainability Platform.
"""

__version__ = "0.1.0"
__author__ = "WhiteBoxAI Team"
__license__ = "MIT"

from explainai.client import WhiteBoxAI
from explainai.decorators import monitor_model, monitor_prediction
from explainai.monitor import ModelMonitor

__all__ = [
    "WhiteBoxAI",
    "ModelMonitor",
    "monitor_model",
    "monitor_prediction",
]
