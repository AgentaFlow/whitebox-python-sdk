"""
Tests for LangChain Integration
"""

from unittest.mock import Mock

import pytest

# The LangChain integration tests exercise the real `langchain` package. Skip
# the whole module when it isn't installed rather than failing on ImportError.
pytest.importorskip("langchain", reason="langchain is an optional dependency")


class TestLangChainMonitor:
    """Test LangChainMonitor class."""

    def test_init(self):
        """Test LangChainMonitor initialization."""
        from whiteboxxai.integrations.langchain import LangChainMonitor

        client = Mock()

        monitor = LangChainMonitor(
            client=client,
            application_name="test_app",
            track_tokens=True,
            track_cost=True,
        )

        assert monitor._application_name == "test_app"
        assert monitor._track_tokens is True
        assert monitor._track_cost is True
        assert not monitor._registered

    def test_init_without_langchain(self, monkeypatch):
        """Test initialization fails without langchain."""
        from whiteboxxai.integrations import langchain as lc_module

        # monkeypatch (not a direct attribute assignment) so LANGCHAIN_AVAILABLE
        # is restored to its real value after this test -- lc_module is a
        # cached singleton, so a plain `lc_module.LANGCHAIN_AVAILABLE = False`
        # would otherwise leak into every subsequent test in the process.
        monkeypatch.setattr(lc_module, "LANGCHAIN_AVAILABLE", False)

        with pytest.raises(ImportError, match="langchain is not installed"):
            lc_module.LangChainMonitor(client=Mock(), application_name="test")

    def test_register_application(self):
        """Test application registration."""
        from whiteboxxai.integrations.langchain import LangChainMonitor

        client = Mock()
        client.models.register = Mock(return_value={"id": 123})

        monitor = LangChainMonitor(client=client, application_name="my_app")

        # Register application
        app_id = monitor.register_application(
            name="Test App", version="1.0.0", description="Test application"
        )

        assert app_id == 123
        assert monitor._registered

        # Verify models.register was called with the fields ModelCreate requires
        client.models.register.assert_called_once()
        call_kwargs = client.models.register.call_args[1]
        assert call_kwargs["name"] == "Test App"
        assert call_kwargs["model_type"] == "llm"
        assert call_kwargs["framework"] == "langchain"
        assert call_kwargs["version"] == "1.0.0"
        assert call_kwargs["description"] == "Test application"
        assert call_kwargs["features"] == ["input"]
        assert call_kwargs["baseline_metrics"] == {}

    def test_register_application_auto_name(self):
        """Test auto-generated application name."""
        from whiteboxxai.integrations.langchain import LangChainMonitor

        client = Mock()
        client.models.register = Mock(return_value={"id": 456})

        monitor = LangChainMonitor(client=client, application_name="custom_name")

        # Register without explicit name
        monitor.register_application()

        call_kwargs = client.models.register.call_args[1]
        assert call_kwargs["name"] == "custom_name"

    def test_create_callback_handler(self):
        """Test callback handler creation."""
        from whiteboxxai.integrations.langchain import LangChainMonitor, WhiteBoxXAICallbackHandler

        client = Mock()
        client.models.register = Mock(return_value={"id": 789})

        monitor = LangChainMonitor(client=client)

        # Create callback (should auto-register)
        callback = monitor.create_callback_handler()

        assert isinstance(callback, WhiteBoxXAICallbackHandler)
        assert monitor._registered
        assert callback.monitor == monitor

    def test_log_chain_execution(self):
        """Test logging chain execution."""
        from whiteboxxai.integrations.langchain import LangChainMonitor

        client = Mock()
        monitor = LangChainMonitor(client=client)
        monitor.log_prediction = Mock()

        # Log chain execution
        monitor.log_chain_execution(
            chain_name="test_chain",
            inputs={"question": "What is AI?"},
            outputs={"answer": "AI is..."},
            execution_time=1.5,
            llm_calls=[{"model": "gpt-3.5", "tokens": 100}],
            tool_calls=[{"tool": "search", "result": "..."}],
        )

        # Verify log_prediction was called
        monitor.log_prediction.assert_called_once()
        call_kwargs = monitor.log_prediction.call_args[1]
        assert call_kwargs["inputs"] == {"question": "What is AI?"}
        assert call_kwargs["output"] == {"answer": "AI is..."}
        assert call_kwargs["metadata"]["chain_name"] == "test_chain"
        assert call_kwargs["metadata"]["execution_time"] == 1.5
        assert call_kwargs["metadata"]["num_llm_calls"] == 1
        assert call_kwargs["metadata"]["num_tool_calls"] == 1

    def test_log_agent_execution(self):
        """Test logging agent execution."""
        from whiteboxxai.integrations.langchain import LangChainMonitor

        client = Mock()
        monitor = LangChainMonitor(client=client)
        monitor.log_prediction = Mock()

        # Log agent execution
        steps = [
            {"tool": "search", "input": "query", "output": "results"},
            {"tool": "calculator", "input": "2+2", "output": "4"},
        ]

        monitor.log_agent_execution(
            agent_name="test_agent",
            inputs={"task": "Solve problem"},
            outputs={"result": "Solution"},
            execution_time=2.5,
            steps=steps,
        )

        # Verify log_prediction was called
        monitor.log_prediction.assert_called_once()
        call_kwargs = monitor.log_prediction.call_args[1]
        assert call_kwargs["metadata"]["agent_name"] == "test_agent"
        assert call_kwargs["metadata"]["execution_time"] == 2.5
        assert call_kwargs["metadata"]["num_steps"] == 2
        assert call_kwargs["metadata"]["steps"] == steps

    def test_log_llm_call(self):
        """log_llm_call() calls the real LLM monitoring endpoint
        (client.llm.log_call), not the generic prediction-logging endpoint
        -- so token/cost/latency data is queryable via /llm/stats instead of
        sitting as opaque metadata on a prediction record."""
        from whiteboxxai.integrations.langchain import LangChainMonitor

        client = Mock()
        monitor = LangChainMonitor(client=client)

        monitor.log_llm_call(
            prompt="What is AI?",
            response="AI is artificial intelligence",
            model="gpt-4",
            provider="openai",
            prompt_tokens=12,
            completion_tokens=150,
            latency=1.2,
        )

        client.llm.log_call.assert_called_once()
        call_kwargs = client.llm.log_call.call_args[1]
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["model_name"] == "gpt-4"
        assert call_kwargs["prompt"] == "What is AI?"
        assert call_kwargs["completion"] == "AI is artificial intelligence"
        assert call_kwargs["prompt_tokens"] == 12
        assert call_kwargs["completion_tokens"] == 150
        assert call_kwargs["latency_ms"] == 1200.0

    def test_log_llm_call_tokens_used_falls_back_to_completion_tokens(self):
        """Legacy tokens_used kwarg still works when completion_tokens isn't
        given separately."""
        from whiteboxxai.integrations.langchain import LangChainMonitor

        client = Mock()
        monitor = LangChainMonitor(client=client)

        monitor.log_llm_call(prompt="hi", response="hello", model="gpt-4", tokens_used=150)

        call_kwargs = client.llm.log_call.call_args[1]
        assert call_kwargs["completion_tokens"] == 150

    def test_log_tool_call(self):
        """Test logging tool call."""
        from whiteboxxai.integrations.langchain import LangChainMonitor

        client = Mock()
        monitor = LangChainMonitor(client=client)
        monitor.log_prediction = Mock()

        # Log tool call
        monitor.log_tool_call(
            tool_name="web_search",
            tool_input="latest news",
            tool_output="Search results...",
            execution_time=0.8,
        )

        # Verify log_prediction was called
        monitor.log_prediction.assert_called_once()
        call_kwargs = monitor.log_prediction.call_args[1]
        assert call_kwargs["inputs"] == {"tool_input": "latest news"}
        assert call_kwargs["output"] == {"tool_output": "Search results..."}
        assert call_kwargs["metadata"]["tool_name"] == "web_search"
        assert call_kwargs["metadata"]["execution_time"] == 0.8

    def test_log_rag_retrieval(self):
        """log_rag_retrieval() calls the real RAG monitoring endpoint
        (client.rag.log_retrieval), not the generic prediction-logging
        endpoint -- so retrieval quality metrics are computed server-side
        and queryable via /rag/stats."""
        from whiteboxxai.integrations.langchain import LangChainMonitor

        client = Mock()
        monitor = LangChainMonitor(client=client)

        documents = [
            {"id": "doc1", "text": "Doc 1", "score": 0.95},
            {"id": "doc2", "text": "Doc 2", "score": 0.85},
        ]

        monitor.log_rag_retrieval(
            query="test query",
            documents=documents,
            num_retrieved=2,
            retrieval_time=0.5,
            ground_truth_ids=["doc1"],
        )

        client.rag.log_retrieval.assert_called_once()
        call_kwargs = client.rag.log_retrieval.call_args[1]
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["top_k"] == 2
        assert call_kwargs["retrieval_latency_ms"] == 500.0
        assert call_kwargs["ground_truth_ids"] == ["doc1"]
        results = call_kwargs["results"]
        assert results[0] == {
            "document_id": "doc1",
            "document_content": "Doc 1",
            "document_metadata": None,
            "rank": 1,
            "score": 0.95,
        }

    def test_log_rag_retrieval_relevance_scores_override_document_score(self):
        from whiteboxxai.integrations.langchain import LangChainMonitor

        client = Mock()
        monitor = LangChainMonitor(client=client)

        monitor.log_rag_retrieval(
            query="q",
            documents=[{"text": "Doc 1", "score": 0.1}],
            num_retrieved=1,
            relevance_scores=[0.99],
        )

        results = client.rag.log_retrieval.call_args[1]["results"]
        assert results[0]["score"] == 0.99


class TestWhiteBoxXAICallbackHandler:
    """Test WhiteBoxXAICallbackHandler class."""

    def test_init(self):
        """Test callback handler initialization."""
        from whiteboxxai.integrations.langchain import LangChainMonitor, WhiteBoxXAICallbackHandler

        monitor = Mock(spec=LangChainMonitor)
        callback = WhiteBoxXAICallbackHandler(monitor=monitor)

        assert callback.monitor == monitor
        assert len(callback._chain_start_times) == 0
        assert len(callback._agent_start_times) == 0

    def test_on_chain_start_end(self):
        """Test chain start/end callbacks."""
        from whiteboxxai.integrations.langchain import LangChainMonitor, WhiteBoxXAICallbackHandler

        monitor = Mock(spec=LangChainMonitor)
        monitor.log_chain_execution = Mock()

        callback = WhiteBoxXAICallbackHandler(monitor=monitor)

        # Start chain
        run_id = "test_run_1"
        inputs = {"question": "Test?"}
        callback.on_chain_start(serialized={}, inputs=inputs, run_id=run_id, name="test_chain")

        assert run_id in callback._chain_start_times
        assert run_id in callback._chain_inputs

        # End chain
        outputs = {"answer": "Answer"}
        callback.on_chain_end(outputs=outputs, run_id=run_id, name="test_chain")

        # Verify logging was called
        monitor.log_chain_execution.assert_called_once()
        call_kwargs = monitor.log_chain_execution.call_args[1]
        assert call_kwargs["chain_name"] == "test_chain"
        assert call_kwargs["inputs"] == inputs
        assert call_kwargs["outputs"] == outputs

        # Verify cleanup
        assert run_id not in callback._chain_start_times
        assert run_id not in callback._chain_inputs

    def test_on_chain_error(self):
        """Test chain error callback."""
        from whiteboxxai.integrations.langchain import LangChainMonitor, WhiteBoxXAICallbackHandler

        monitor = Mock(spec=LangChainMonitor)
        callback = WhiteBoxXAICallbackHandler(monitor=monitor)

        # Start chain
        run_id = "error_run"
        callback.on_chain_start(serialized={}, inputs={}, run_id=run_id)

        # Error occurs
        callback.on_chain_error(error=Exception("Test error"), run_id=run_id)

        # Verify cleanup
        assert run_id not in callback._chain_start_times

    def test_on_tool_start_end(self):
        """Test tool start/end callbacks."""
        from whiteboxxai.integrations.langchain import LangChainMonitor, WhiteBoxXAICallbackHandler

        monitor = Mock(spec=LangChainMonitor)
        monitor.log_tool_call = Mock()

        callback = WhiteBoxXAICallbackHandler(monitor=monitor)

        # Start tool
        run_id = "tool_run"
        callback.on_tool_start(
            serialized={}, input_str="test input", run_id=run_id, name="test_tool"
        )

        assert run_id in callback._tool_start_times

        # End tool
        callback.on_tool_end(
            output="test output",
            run_id=run_id,
            name="test_tool",
            input_str="test input",
        )

        # Verify logging was called
        monitor.log_tool_call.assert_called_once()

        # Verify cleanup
        assert run_id not in callback._tool_start_times


class TestWrapLangChainChain:
    """Test wrap_langchain_chain function."""

    def test_wrap_chain(self):
        """Test wrapping a chain."""
        from whiteboxxai.integrations.langchain import LangChainMonitor, wrap_langchain_chain

        # Mock chain
        chain = Mock()
        original_run = Mock(return_value="result")
        chain.run = original_run
        chain.__call__ = Mock(return_value={"output": "result"})

        # Mock monitor
        monitor = Mock(spec=LangChainMonitor)
        monitor._registered = True
        monitor.create_callback_handler = Mock(return_value=Mock())

        # Wrap chain
        wrapped = wrap_langchain_chain(chain, monitor)

        # Call wrapped chain
        wrapped.run(input="test")

        # Verify original run was called
        assert original_run.called

        # Verify callback was added
        call_kwargs = original_run.call_args[1]
        assert "callbacks" in call_kwargs
        assert len(call_kwargs["callbacks"]) > 0

    def test_wrap_chain_auto_register(self):
        """Test wrapping auto-registers if needed."""
        from whiteboxxai.integrations.langchain import LangChainMonitor, wrap_langchain_chain

        chain = Mock()
        chain.run = Mock(return_value="result")

        monitor = Mock(spec=LangChainMonitor)
        monitor._registered = False
        monitor.register_application = Mock()
        monitor.create_callback_handler = Mock(return_value=Mock())

        # Wrap chain (should auto-register)
        wrap_langchain_chain(chain, monitor)

        # Verify registration was called
        monitor.register_application.assert_called_once()


class TestEndToEndLangChain:
    """End-to-end integration test."""

    def test_full_workflow(self):
        """Test complete workflow with mocked LangChain."""
        from whiteboxxai.integrations.langchain import LangChainMonitor

        # Mock client. Not spec'd against WhiteBoxXAI: `.models`/`.predictions`
        # are instance attributes set in WhiteBoxXAI.__init__, so a class-level
        # spec wouldn't expose them and real log_prediction()/register_model()
        # calls would raise AttributeError.
        client = Mock()
        client.models.register = Mock(return_value={"id": 999})
        client.log_prediction = Mock()

        # Create monitor
        monitor = LangChainMonitor(
            client=client,
            application_name="test_app",
            track_tokens=True,
            track_cost=True,
        )

        # Register application
        app_id = monitor.register_application(name="Test LangChain App", version="1.0.0")

        assert app_id == 999

        # Create callback handler
        callback = monitor.create_callback_handler()
        assert callback is not None

        # Log chain execution
        monitor.log_chain_execution(
            chain_name="qa_chain",
            inputs={"question": "What is AI?"},
            outputs={"answer": "AI is..."},
            execution_time=1.5,
        )

        # Log LLM call
        monitor.log_llm_call(
            prompt="Explain AI",
            response="AI explanation...",
            model="gpt-3.5-turbo",
            tokens_used=100,
            latency=0.8,
        )

        # Log tool call
        monitor.log_tool_call(
            tool_name="search",
            tool_input="AI research",
            tool_output="Results...",
            execution_time=0.5,
        )

        # Log RAG retrieval
        monitor.log_rag_retrieval(
            query="What is machine learning?",
            documents=[{"text": "ML is...", "score": 0.9}],
            num_retrieved=1,
            retrieval_time=0.3,
        )
