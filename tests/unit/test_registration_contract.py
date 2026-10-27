"""
Contract tests for the integration helpers' model registration.

The helpers used to ship sending payloads the API rejects (``model_type`` is a
strict enum, ``version``/``features``/``baseline_metrics`` are required) and
calling methods that don't exist on ``WhiteBoxXAI``
(``client.register_model``). Their unit tests never noticed: they handed the
helper a bare ``Mock()`` client, which accepts any call, and asserted on the
(wrong) call the helper made.

These tests close that gap two ways:

* every ``register_from_model`` / ``register_application`` / ``register_saved_model``
  is driven through the real ``ModelsResource``, and the request body it would
  POST is validated against the backend's real ``ModelCreate`` schema (or, where
  the backend isn't importable -- the public SDK repo -- against ``_ModelCreateMirror``,
  a copy of the fields and enum that schema requires);
* a static check that every ``client.<...>`` attribute chain the integrations
  use exists on a real ``WhiteBoxXAI`` client.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

import whiteboxxai
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations._registration import (
    TASK_NAMES,
    VALID_MODEL_TYPES,
    registration_fields,
)
from whiteboxxai.resources import ModelsResource

INTEGRATIONS_DIR = Path(whiteboxxai.__file__).parent / "integrations"


class RecordingClient:
    """A WhiteBoxXAI stand-in whose HTTP layer records what would be sent.

    ``models`` is the real ModelsResource, so ``models.register()`` builds the
    exact body it would POST to ``/api/v1/models/``.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, Any]] = []
        self.models = ModelsResource(self)

    def request(self, method, endpoint, data=None, params=None):
        self.requests.append((method, endpoint, data))
        return {"id": "model-1"}

    def is_offline_enabled(self) -> bool:
        return False

    @property
    def registered(self) -> dict[str, Any]:
        """The body of the last POST /api/v1/models/."""
        bodies = [
            data
            for method, endpoint, data in self.requests
            if method == "POST" and endpoint == "/api/v1/models/"
        ]
        assert bodies, "nothing was registered"
        return bodies[-1]


@pytest.fixture
def client() -> RecordingClient:
    return RecordingClient()


class _ModelCreateMirror:
    """The part of the backend's ``ModelCreate`` this SDK has to satisfy.

    Used only where ``backend`` can't be imported. It mirrors backend/schemas/model.py:
    ``name``, ``version``, ``model_type`` (strict ``ModelType`` enum), ``features`` and
    ``baseline_metrics`` are required. When the backend is importable the tests use the
    real schema instead, and ``test_valid_model_types_match_the_backend_enum`` fails if
    this mirror's enum drifts from it.
    """

    REQUIRED = ("name", "version", "model_type", "features", "baseline_metrics")

    @classmethod
    def model_validate(cls, body):
        missing = [field for field in cls.REQUIRED if body.get(field) is None]
        if missing:
            raise ValueError(f"ModelCreate rejects the request: missing {missing}")
        if body["model_type"] not in VALID_MODEL_TYPES:
            raise ValueError(f"ModelCreate rejects model_type {body['model_type']!r}")
        if not isinstance(body["features"], list):
            raise ValueError("ModelCreate rejects non-list features")
        if not isinstance(body["baseline_metrics"], dict):
            raise ValueError("ModelCreate rejects non-dict baseline_metrics")
        parsed = dict(body, model_type=SimpleNamespace(value=body["model_type"]))
        return SimpleNamespace(**parsed)


@pytest.fixture(scope="module")
def model_create():
    """The backend's request schema, or its mirror where the backend isn't importable."""
    try:
        from backend.schemas.model import ModelCreate
    except ImportError:
        return _ModelCreateMirror
    return ModelCreate


def validated(client: RecordingClient, model_create):
    """Validate the registered body against ModelCreate (raises on a 422)."""
    return model_create.model_validate(client.registered)


# --------------------------------------------------------------------------
# The shared rules
# --------------------------------------------------------------------------


class TestRegistrationFields:
    def test_valid_model_types_match_the_backend_enum(self):
        database = pytest.importorskip(
            "backend.models.database", reason="backend package not importable"
        )
        ModelType = database.ModelType

        assert set(VALID_MODEL_TYPES) == {member.value for member in ModelType}

    def test_a_valid_model_type_is_used_as_given(self):
        fields = registration_fields(
            model_type="llm",
            default_model_type="custom",
            version=None,
            features=["a"],
            inferred_features=None,
            baseline_metrics=None,
            metadata={},
        )
        assert fields["model_type"] == "llm"

    @pytest.mark.parametrize("task", sorted(TASK_NAMES))
    def test_a_task_name_becomes_metadata_not_model_type(self, task):
        metadata: dict[str, Any] = {}
        fields = registration_fields(
            model_type=task,
            default_model_type="neural_network",
            version="1.0.0",
            features=["a"],
            inferred_features=None,
            baseline_metrics=None,
            metadata=metadata,
        )
        assert fields["model_type"] == "neural_network"
        assert metadata["task"] == task

    def test_an_unknown_model_type_is_rejected_before_any_request(self):
        with pytest.raises(ValueError, match="Invalid model_type 'banana'"):
            registration_fields(
                model_type="banana",
                default_model_type="custom",
                version="1.0.0",
                features=["a"],
                inferred_features=None,
                baseline_metrics=None,
                metadata={},
            )

    def test_missing_features_ask_for_them(self):
        with pytest.raises(ValueError, match="features"):
            registration_fields(
                model_type=None,
                default_model_type="custom",
                version="1.0.0",
                features=None,
                inferred_features=None,
                baseline_metrics=None,
                metadata={},
            )

    def test_explicit_features_win_over_inferred(self):
        fields = registration_fields(
            model_type=None,
            default_model_type="custom",
            version="1.0.0",
            features=["mine"],
            inferred_features=["inferred"],
            baseline_metrics={"auc": 0.9},
            metadata={},
        )
        assert fields["features"] == ["mine"]
        assert fields["baseline_metrics"] == {"auc": 0.9}


# --------------------------------------------------------------------------
# scikit-learn
# --------------------------------------------------------------------------


class TestSklearnRegistration:
    def test_random_forest_with_named_columns(self, client, model_create):
        pd = pytest.importorskip("pandas")
        ensemble = pytest.importorskip("sklearn.ensemble")
        from whiteboxxai.integrations.sklearn import SklearnMonitor

        X = pd.DataFrame(np.random.rand(20, 3), columns=["age", "income", "tenure"])
        model = ensemble.RandomForestClassifier(n_estimators=3).fit(X, [0, 1] * 10)

        SklearnMonitor(client, model=model).register_from_model(
            version="2.1.0",
            baseline_metrics={"accuracy": 0.9},
            model_type="classification",  # a task name still works
        )

        parsed = validated(client, model_create)
        assert parsed.model_type.value == "random_forest"
        assert parsed.features == ["age", "income", "tenure"]
        assert parsed.version == "2.1.0"
        assert parsed.baseline_metrics == {"accuracy": 0.9}
        assert client.registered["task"] == "classification"

    def test_unnamed_columns_get_positional_names(self, client, model_create):
        linear = pytest.importorskip("sklearn.linear_model")
        from whiteboxxai.integrations.sklearn import SklearnMonitor

        model = linear.LogisticRegression().fit(np.random.rand(20, 4), [0, 1] * 10)

        SklearnMonitor(client, model=model).register_from_model()

        parsed = validated(client, model_create)
        assert parsed.model_type.value == "linear_model"
        assert parsed.features == ["feature_0", "feature_1", "feature_2", "feature_3"]

    def test_an_estimator_with_no_matching_type_is_custom(self, client, model_create):
        neighbors = pytest.importorskip("sklearn.neighbors")
        from whiteboxxai.integrations.sklearn import SklearnMonitor

        model = neighbors.KNeighborsClassifier(n_neighbors=1).fit(
            np.random.rand(5, 2), [0, 1, 0, 1, 0]
        )

        SklearnMonitor(client, model=model).register_from_model()

        assert validated(client, model_create).model_type.value == "custom"

    def test_an_unfitted_estimator_needs_features(self, client):
        ensemble = pytest.importorskip("sklearn.ensemble")
        from whiteboxxai.integrations.sklearn import SklearnMonitor

        monitor = SklearnMonitor(client, model=ensemble.RandomForestClassifier())

        with pytest.raises(ValueError, match="features"):
            monitor.register_from_model()
        assert client.requests == []

        monitor.register_from_model(features=["a", "b"])
        assert client.registered["features"] == ["a", "b"]


# --------------------------------------------------------------------------
# PyTorch
# --------------------------------------------------------------------------


class TestTorchRegistration:
    def test_features_come_from_the_first_linear_layer(self, client, model_create):
        torch = pytest.importorskip("torch")
        from whiteboxxai.integrations.pytorch import TorchMonitor

        net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))

        TorchMonitor(client, model=net).register_from_model(name="Torch Net")

        parsed = validated(client, model_create)
        assert parsed.model_type.value == "neural_network"
        assert parsed.features == ["feature_0", "feature_1", "feature_2", "feature_3"]

    def test_a_module_without_a_linear_layer_needs_features(self, client):
        torch = pytest.importorskip("torch")
        from whiteboxxai.integrations.pytorch import TorchMonitor

        monitor = TorchMonitor(client, model=torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3)))

        with pytest.raises(ValueError, match="features"):
            monitor.register_from_model(name="Conv Net")


# --------------------------------------------------------------------------
# TensorFlow / Keras
# --------------------------------------------------------------------------


class TestKerasRegistration:
    @pytest.fixture
    def keras_model(self):
        tf = pytest.importorskip("tensorflow")
        keras = tf.keras
        return keras.Sequential(
            [
                keras.layers.Input(shape=(6,)),
                keras.layers.Dense(4),
                keras.layers.Dense(1),
            ]
        )

    def test_register_from_model(self, client, model_create, keras_model):
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        KerasMonitor(
            client,
            model=keras_model,
            model_name="keras_net",
            model_type="classification",
        ).register_from_model()

        parsed = validated(client, model_create)
        assert parsed.model_type.value == "neural_network"
        assert parsed.features == [f"feature_{i}" for i in range(6)]
        assert client.registered["metadata"]["task"] == "classification"

    def test_register_saved_model_no_longer_calls_a_missing_client_method(
        self, client, model_create
    ):
        pytest.importorskip("tensorflow")
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        KerasMonitor(client, model=None, model_name="saved_net").register_saved_model(
            "saved_models/net", features=["x1", "x2"]
        )

        parsed = validated(client, model_create)
        assert parsed.features == ["x1", "x2"]
        assert client.registered["metadata"]["format"] == "SavedModel"


# --------------------------------------------------------------------------
# XGBoost / LightGBM
# --------------------------------------------------------------------------


@pytest.fixture
def frame():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(np.random.rand(30, 4), columns=["f1", "f2", "f3", "f4"]), np.array(
        [0, 1] * 15
    )


class TestBoostingRegistration:
    def test_xgboost(self, client, model_create, frame):
        xgb = pytest.importorskip("xgboost")
        from whiteboxxai.integrations.boosting import XGBoostMonitor

        X, y = frame
        model = xgb.XGBClassifier(n_estimators=2).fit(X, y)

        XGBoostMonitor(client, model_name="xgb_model").register_from_model(model, version="3.0.0")

        parsed = validated(client, model_create)
        assert parsed.model_type.value == "xgboost"
        assert parsed.features == ["f1", "f2", "f3", "f4"]
        assert parsed.version == "3.0.0"
        assert client.registered["metadata"]["task"] == "classification"

    def test_xgboost_auto_register_through_wrap(self, client, model_create, frame):
        xgb = pytest.importorskip("xgboost")
        from whiteboxxai.integrations.boosting import XGBoostMonitor, wrap_xgboost_model

        X, y = frame
        model = xgb.XGBRegressor(n_estimators=2).fit(X, y)

        wrap_xgboost_model(model, XGBoostMonitor(client, model_name="xgb_wrapped"))

        parsed = validated(client, model_create)
        assert parsed.model_type.value == "xgboost"
        assert client.registered["metadata"]["task"] == "regression"

    def test_lightgbm(self, client, model_create, frame):
        lgb = pytest.importorskip("lightgbm")
        from whiteboxxai.integrations.boosting import LightGBMMonitor

        X, y = frame
        model = lgb.LGBMClassifier(n_estimators=2, verbose=-1).fit(X, y)

        LightGBMMonitor(client, model_name="lgb_model").register_from_model(model)

        parsed = validated(client, model_create)
        # The API has no LightGBM member in its model-type enum
        assert parsed.model_type.value == "custom"
        assert parsed.features == ["f1", "f2", "f3", "f4"]
        assert client.registered["model_framework"] == "lightgbm"

    def test_caller_can_override_model_type(self, client, model_create, frame):
        lgb = pytest.importorskip("lightgbm")
        from whiteboxxai.integrations.boosting import LightGBMMonitor

        X, y = frame
        model = lgb.LGBMClassifier(n_estimators=2, verbose=-1).fit(X, y)

        LightGBMMonitor(client, model_name="lgb_model").register_from_model(
            model, model_type="xgboost"
        )

        assert validated(client, model_create).model_type.value == "xgboost"


# --------------------------------------------------------------------------
# Hugging Face Transformers
# --------------------------------------------------------------------------


class TestTransformersRegistration:
    @staticmethod
    def _monitor(client, task):
        from whiteboxxai.integrations import transformers as trans_module

        if not trans_module.TRANSFORMERS_AVAILABLE:
            pytest.skip("transformers is not installed")

        pipeline = Mock()
        pipeline.task = task
        pipeline.model.config = Mock(model_type="bert", num_parameters=1, vocab_size=2)
        return trans_module.TransformersMonitor(client, pipeline=pipeline, model_name="hf_model")

    @pytest.mark.parametrize(
        "task, expected",
        [
            ("text-classification", "neural_network"),
            ("question-answering", "neural_network"),
            ("text-generation", "llm"),
            ("summarization", "llm"),
            ("translation_en_to_fr", "llm"),
        ],
    )
    def test_task_maps_to_a_valid_model_type(self, client, model_create, task, expected):
        monitor = self._monitor(client, task)

        with patch("whiteboxxai.integrations.transformers.transformers") as mock_tf:
            mock_tf.__version__ = "4.30.0"
            monitor.register_from_model()

        parsed = validated(client, model_create)
        assert parsed.model_type.value == expected
        assert parsed.features == ["text"]
        assert client.registered["metadata"]["task"] == task

    def test_caller_can_override_model_type(self, client, model_create):
        monitor = self._monitor(client, "text-classification")

        with patch("whiteboxxai.integrations.transformers.transformers") as mock_tf:
            mock_tf.__version__ = "4.30.0"
            monitor.register_from_model(model_type="custom")

        assert validated(client, model_create).model_type.value == "custom"


# --------------------------------------------------------------------------
# LangChain
# --------------------------------------------------------------------------


class TestLangChainRegistration:
    @pytest.fixture
    def monitor(self, client, monkeypatch):
        from whiteboxxai.integrations import langchain as lc_module

        # The registration path needs no LangChain types; don't require the
        # package just to test it.
        monkeypatch.setattr(lc_module, "LANGCHAIN_AVAILABLE", True)
        return lc_module.LangChainMonitor(client, application_name="my_app")

    def test_register_application(self, monitor, client, model_create):
        app_id = monitor.register_application(version="1.0.0", description="QA bot")

        parsed = validated(client, model_create)
        assert parsed.model_type.value == "llm"
        assert parsed.features == ["input"]
        assert parsed.description == "QA bot"
        assert app_id == "model-1"
        assert monitor.model_id == "model-1"

    def test_create_callback_handler_registers_instead_of_raising(
        self, monitor, client, model_create
    ):
        """The documented setup (``monitor.create_callback_handler()``) used to
        raise AttributeError: it registered through client.register_model."""
        handler = monitor.create_callback_handler()

        assert handler.monitor is monitor
        assert monitor._registered
        validated(client, model_create)


# --------------------------------------------------------------------------
# Static guard: integrations may only call things the client really has
# --------------------------------------------------------------------------


def _client_attribute_chains(source: str) -> set[tuple[str, ...]]:
    """Every ``client.a`` / ``client.a.b`` / ``self.client.a.b`` chain in code."""
    chains: set[tuple[str, ...]] = set()

    def root_and_names(node: ast.AST):
        names: list[str] = []
        while isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        names.reverse()
        return node, names

    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Attribute):
            continue
        root, names = root_and_names(node)
        if not isinstance(root, ast.Name):
            continue
        if root.id == "client":
            chain = tuple(names)
        elif root.id == "self" and names[:1] == ["client"]:
            chain = tuple(names[1:])
        else:
            continue
        if chain:
            chains.add(chain)
    return chains


class TestIntegrationsOnlyCallRealClientMethods:
    @pytest.mark.parametrize("path", sorted(INTEGRATIONS_DIR.glob("*.py")), ids=lambda p: p.name)
    def test_every_client_attribute_chain_exists(self, path):
        real_client = WhiteBoxXAI(api_key="test_key")
        missing = []
        for chain in sorted(_client_attribute_chains(path.read_text(encoding="utf-8"))):
            target: Any = real_client
            for name in chain[:2]:  # client.<resource>.<method>
                if not hasattr(target, name):
                    missing.append("client." + ".".join(chain))
                    break
                target = getattr(target, name)

        assert not missing, f"{path.name} uses client methods that do not exist: {missing}"
