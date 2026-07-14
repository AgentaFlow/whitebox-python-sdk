"""Framework integrations for WhiteBoxXAI SDK."""

# Scikit-learn integration
try:
    from .sklearn import SklearnMonitor, SklearnWrapper

    __all__ = ["SklearnMonitor", "SklearnWrapper"]
except ImportError:
    pass

# PyTorch integration
try:
    from .pytorch import TorchMonitor, TorchWrapper

    if "__all__" in dir():
        __all__.extend(["TorchMonitor", "TorchWrapper"])
    else:
        __all__ = ["TorchMonitor", "TorchWrapper"]
except ImportError:
    pass

# TensorFlow/Keras integration
try:
    from .tensorflow import KerasMonitor, WhiteBoxXAICallback, wrap_keras_model

    if "__all__" in dir():
        __all__.extend(["KerasMonitor", "WhiteBoxXAICallback", "wrap_keras_model"])
    else:
        __all__ = ["KerasMonitor", "WhiteBoxXAICallback", "wrap_keras_model"]
except ImportError:
    pass

# Hugging Face Transformers integration
try:
    from .transformers import (
        TransformersMonitor,
        TransformersPipelineWrapper,
        wrap_transformers_pipeline,
    )

    if "__all__" in dir():
        __all__.extend(
            [
                "TransformersMonitor",
                "TransformersPipelineWrapper",
                "wrap_transformers_pipeline",
            ]
        )
    else:
        __all__ = [
            "TransformersMonitor",
            "TransformersPipelineWrapper",
            "wrap_transformers_pipeline",
        ]
except ImportError:
    pass

# LangChain integration
try:
    from .langchain import (
        LangChainMonitor,
        WhiteBoxXAICallbackHandler,
        wrap_langchain_chain,
    )

    if "__all__" in dir():
        __all__.extend(
            ["LangChainMonitor", "WhiteBoxXAICallbackHandler", "wrap_langchain_chain"]
        )
    else:
        __all__ = [
            "LangChainMonitor",
            "WhiteBoxXAICallbackHandler",
            "wrap_langchain_chain",
        ]
except ImportError:
    pass

# XGBoost/LightGBM integration
try:
    from .boosting import (
        LightGBMMonitor,
        XGBoostMonitor,
        wrap_lightgbm_model,
        wrap_xgboost_model,
    )

    if "__all__" in dir():
        __all__.extend(
            [
                "XGBoostMonitor",
                "LightGBMMonitor",
                "wrap_xgboost_model",
                "wrap_lightgbm_model",
            ]
        )
    else:
        __all__ = [
            "XGBoostMonitor",
            "LightGBMMonitor",
            "wrap_xgboost_model",
            "wrap_lightgbm_model",
        ]
except ImportError:
    pass

# CrewAI integration
try:
    from .crewai_monitor import CrewAIMonitor, monitor_crew

    if "__all__" in dir():
        __all__.extend(["CrewAIMonitor", "monitor_crew"])
    else:
        __all__ = ["CrewAIMonitor", "monitor_crew"]
except ImportError:
    pass

# LangChain integration
try:
    from .langchain_agents import (
        LangGraphMultiAgentMonitor,
        MultiAgentCallbackHandler,
        monitor_langchain_agent,
    )

    if "__all__" in dir():
        __all__.extend(
            [
                "MultiAgentCallbackHandler",
                "LangGraphMultiAgentMonitor",
                "monitor_langchain_agent",
            ]
        )
    else:
        __all__ = [
            "MultiAgentCallbackHandler",
            "LangGraphMultiAgentMonitor",
            "monitor_langchain_agent",
        ]
except ImportError:
    pass

# Ensure __all__ exists even if all imports fail
if "__all__" not in dir():
    __all__ = []
