"""Unit tests for LangChain multi-agent integration."""

import logging
from unittest.mock import Mock, create_autospec

import pytest

# NOTE: this requires langchain_core (preferred) or the real legacy
# `langchain` package (see the `langchain` extra in sdk/pyproject.toml) to be
# installed. langchain_agents.py subclasses the real BaseCallbackHandler at
# class-definition time, so faking out sys.modules["langchain.callbacks.base"]
# etc. here (as this file previously did) makes that inheritance operate on a
# Mock instead of a real class, producing undefined/garbage behavior. Skip
# cleanly (rather than erroring at collection) in environments -- like the
# root "Unit Tests" CI job, which installs requirements-dev.txt and does not
# pull in the SDK's langchain extra -- where neither is importable.
pytest.importorskip("langchain_core", reason="langchain_core is an optional dependency")

from whiteboxxai.integrations.langchain_agents import (  # noqa: E402
    LangGraphMultiAgentMonitor,
    MultiAgentCallbackHandler,
    monitor_langchain_agent,
)
from whiteboxxai.resources import AgentWorkflowsResource  # noqa: E402


@pytest.fixture
def workflows():
    """The real AgentWorkflowsResource surface, with canned responses.

    create_autospec (not a bare Mock) so a call to a method the resource
    doesn't have, or with arguments its signature doesn't accept, raises --
    a bare Mock accepts anything, which is how this integration shipped
    calling create()/create_interaction()/get_analytics() and
    create_execution(agent_name=..., status=...), none of which exist.
    """
    resource = create_autospec(AgentWorkflowsResource, instance=True)
    resource.create_and_start.return_value = {"id": "workflow_123", "status": "pending"}
    resource.register_agent.return_value = {"id": "agent_1"}
    resource.create_execution.return_value = {"id": "exec_123"}
    resource.log_interaction.return_value = {"id": "interaction_1"}
    resource.analytics.return_value = {"total_cost": 0.15, "total_tokens": 750}
    return resource


@pytest.fixture
def mock_client(workflows):
    """A mock WhiteBoxXAI client exposing the autospec'd agent_workflows."""
    client = Mock()
    client.agent_workflows = workflows
    return client


@pytest.fixture
def callback_handler(mock_client):
    """Create a MultiAgentCallbackHandler instance."""
    return MultiAgentCallbackHandler(
        client=mock_client,
        workflow_id="workflow_123",
        agent_name="test_agent",
        agent_role="Test Agent",
    )


class TestMultiAgentCallbackHandler:
    """Tests for MultiAgentCallbackHandler."""

    def test_init(self, mock_client):
        """Test callback handler initialization."""
        handler = MultiAgentCallbackHandler(
            client=mock_client,
            workflow_id="workflow_123",
            agent_name="test_agent",
            agent_role="Test Role",
        )

        assert handler.client == mock_client
        assert handler.workflow_id == "workflow_123"
        assert handler.agent_name == "test_agent"
        assert handler.agent_role == "Test Role"
        assert handler.agent_id is None
        assert handler.track_tokens is True
        assert handler.track_costs is True
        assert handler.llm_call_count == 0
        assert handler.tool_call_count == 0

    def test_on_chain_start_opens_an_execution(self, callback_handler, workflows):
        """A top-level chain run registers the agent and opens an execution."""
        inputs = {"input": "test query"}

        callback_handler.on_chain_start(serialized={"name": "test_chain"}, inputs=inputs)

        workflows.register_agent.assert_called_once_with(
            "workflow_123", name="test_agent", role="Test Agent"
        )
        workflows.create_execution.assert_called_once_with("workflow_123", "agent_1", inputs=inputs)
        assert callback_handler.agent_id == "agent_1"
        assert callback_handler.current_execution_id == "exec_123"
        assert callback_handler.execution_inputs == inputs
        assert callback_handler.execution_start_time is not None
        assert callback_handler.llm_call_count == 0
        assert callback_handler.tool_call_count == 0

    def test_known_agent_id_is_not_registered_again(self, mock_client, workflows):
        """An agent_id from register_agent() is used as given."""
        handler = MultiAgentCallbackHandler(
            client=mock_client,
            workflow_id="workflow_123",
            agent_name="test_agent",
            agent_id="agent_42",
        )

        handler.on_chain_start(serialized={}, inputs={"input": "q"})

        workflows.register_agent.assert_not_called()
        workflows.create_execution.assert_called_once_with(
            "workflow_123", "agent_42", inputs={"input": "q"}
        )

    def test_nested_chain_is_not_a_separate_execution(self, callback_handler, workflows):
        """Only the top-level chain run (no parent_run_id) is an execution."""
        callback_handler.on_chain_start(
            serialized={}, inputs={"input": "q"}, parent_run_id="run_parent"
        )
        callback_handler.on_chain_end(outputs={}, parent_run_id="run_parent")

        workflows.create_execution.assert_not_called()
        workflows.log_interaction.assert_not_called()
        assert callback_handler.execution_start_time is None

    def test_on_chain_end_logs_how_the_execution_ended(self, callback_handler, workflows):
        """The outcome is logged as an execution_end interaction."""
        callback_handler.on_chain_start(serialized={}, inputs={"input": "test"})
        callback_handler.llm_call_count = 2
        callback_handler.tool_call_count = 1
        callback_handler.total_tokens = 150
        callback_handler.total_cost = 0.003

        outputs = {"output": "test result"}
        callback_handler.on_chain_end(outputs=outputs)

        workflows.log_interaction.assert_called_once()
        call_args = workflows.log_interaction.call_args
        assert call_args.args == ("workflow_123",)
        assert call_args.kwargs["interaction_type"] == "execution_end"
        assert call_args.kwargs["from_agent_id"] == "agent_1"
        assert call_args.kwargs["from_execution_id"] == "exec_123"
        meta = call_args.kwargs["meta_data"]
        assert meta["status"] == "completed"
        assert meta["outputs"] == outputs
        assert meta["llm_call_count"] == 2
        assert meta["tool_call_count"] == 1
        assert meta["tokens_used"] == 150
        assert meta["cost"] == 0.003
        assert meta["duration_ms"] >= 0

        # Verify state was reset
        assert callback_handler.execution_start_time is None
        assert callback_handler.current_execution_id is None

    def test_on_chain_error_logs_a_failed_execution(self, callback_handler, workflows):
        """A failed chain run is logged with the error."""
        callback_handler.on_chain_start(serialized={}, inputs={"input": "test"})

        callback_handler.on_chain_error(error=ValueError("Test error"))

        call_args = workflows.log_interaction.call_args
        assert call_args.kwargs["interaction_type"] == "execution_end"
        meta = call_args.kwargs["meta_data"]
        assert meta["status"] == "failed"
        assert "Test error" in meta["error"]

    def test_on_llm_start(self, callback_handler):
        """Test LLM start callback."""
        initial_count = callback_handler.llm_call_count

        callback_handler.on_llm_start(serialized={"name": "test_llm"}, prompts=["test prompt"])

        assert callback_handler.llm_call_count == initial_count + 1

    def test_on_llm_end_with_tokens(self, callback_handler):
        """Test LLM end callback with token tracking."""
        # Mock LLMResult with token usage
        mock_result = Mock()
        mock_result.llm_output = {
            "token_usage": {
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50,
            }
        }

        callback_handler.on_llm_end(response=mock_result)

        assert callback_handler.total_tokens == 100
        assert callback_handler.total_cost > 0  # Should estimate cost

    def test_on_agent_action(self, callback_handler, workflows):
        """Test agent action (tool call) callback."""
        mock_action = Mock()
        mock_action.tool = "search"
        mock_action.tool_input = "AI safety"
        mock_action.log = "Searching for AI safety"

        callback_handler.on_agent_action(action=mock_action)

        workflows.log_interaction.assert_called_once()
        call_args = workflows.log_interaction.call_args
        assert call_args.args == ("workflow_123",)
        assert call_args.kwargs["interaction_type"] == "tool_call"
        assert call_args.kwargs["from_agent_id"] == "agent_1"
        assert "search" in call_args.kwargs["message"]
        assert call_args.kwargs["meta_data"]["tool"] == "search"

        # Verify tool count incremented
        assert callback_handler.tool_call_count == 1

    def test_on_tool_end(self, callback_handler, workflows):
        """Test tool end callback."""
        output = "Search results for AI safety"

        callback_handler.on_tool_end(output=output)

        workflows.log_interaction.assert_called_once()
        call_args = workflows.log_interaction.call_args
        assert call_args.kwargs["interaction_type"] == "tool_result"
        assert call_args.kwargs["from_agent_id"] == "agent_1"
        assert output in call_args.kwargs["message"]

    def test_on_tool_error(self, callback_handler, workflows):
        """Test tool error callback."""
        callback_handler.on_tool_error(error=RuntimeError("Tool failed"))

        workflows.log_interaction.assert_called_once()
        call_args = workflows.log_interaction.call_args
        assert call_args.kwargs["interaction_type"] == "tool_result"
        assert "Tool error" in call_args.kwargs["message"]
        assert "RuntimeError" in call_args.kwargs["meta_data"]["error_type"]


class TestLangGraphMultiAgentMonitor:
    """Tests for LangGraphMultiAgentMonitor."""

    def test_init(self, mock_client):
        """Test monitor initialization."""
        monitor = LangGraphMultiAgentMonitor(
            client=mock_client,
            workflow_name="Test Workflow",
            meta_data={"key": "value"},
        )

        assert monitor.client == mock_client
        assert monitor.workflow_name == "Test Workflow"
        assert monitor.workflow_meta_data == {"key": "value"}
        assert monitor.workflow_id is None
        assert len(monitor.callbacks) == 0

    def test_start_monitoring(self, mock_client, workflows):
        """Test starting workflow monitoring."""
        monitor = LangGraphMultiAgentMonitor(client=mock_client, workflow_name="Test Workflow")

        inputs = {"query": "test"}
        workflow_id = monitor.start_monitoring(inputs=inputs)

        workflows.create_and_start.assert_called_once_with(
            name="Test Workflow", framework="langchain", meta_data={}
        )
        workflows.start.assert_called_once_with("workflow_123", inputs=inputs)

        assert workflow_id == "workflow_123"
        assert monitor.workflow_id == "workflow_123"
        assert monitor.start_time is not None

    def test_register_agent(self, mock_client, workflows):
        """Test agent registration."""
        monitor = LangGraphMultiAgentMonitor(client=mock_client, workflow_name="Test Workflow")
        monitor.start_monitoring()

        agent_id = monitor.register_agent(
            agent_name="researcher",
            role="Research Agent",
            model_name="gpt-4",
            tools=["search", "scrape"],
        )

        workflows.register_agent.assert_called_once_with(
            workflow_id="workflow_123",
            name="researcher",
            role="Research Agent",
            model_name="gpt-4",
            tools=["search", "scrape"],
        )
        assert agent_id == "agent_1"
        assert monitor.agent_ids == {"researcher": "agent_1"}

    def test_get_callbacks(self, mock_client):
        """Test getting callbacks for an agent."""
        monitor = LangGraphMultiAgentMonitor(client=mock_client, workflow_name="Test Workflow")
        monitor.start_monitoring()
        monitor.register_agent("researcher")

        callbacks = monitor.get_callbacks("researcher", agent_role="Research Agent")

        assert len(callbacks) == 1
        assert isinstance(callbacks[0], MultiAgentCallbackHandler)
        assert callbacks[0].agent_name == "researcher"
        assert callbacks[0].agent_role == "Research Agent"
        # The handler reuses the registered agent rather than registering again
        assert callbacks[0].agent_id == "agent_1"

        # Getting callbacks again should return the same instance
        callbacks2 = monitor.get_callbacks("researcher")
        assert callbacks2[0] is callbacks[0]

    def test_log_handoff(self, mock_client, workflows):
        """Test logging agent-to-agent handoff."""
        workflows.register_agent.side_effect = [
            {"id": "agent_supervisor"},
            {"id": "agent_researcher"},
        ]
        monitor = LangGraphMultiAgentMonitor(client=mock_client, workflow_name="Test Workflow")
        monitor.start_monitoring()

        monitor.log_handoff(
            from_agent="supervisor",
            to_agent="researcher",
            message="Please research this topic",
            meta_data={"topic": "AI safety"},
        )

        # Agents the handoff names but that weren't registered yet are registered
        assert workflows.register_agent.call_count == 2
        workflows.log_interaction.assert_called_once_with(
            workflow_id="workflow_123",
            interaction_type="handoff",
            from_agent_id="agent_supervisor",
            to_agent_id="agent_researcher",
            message="Please research this topic",
            meta_data={"topic": "AI safety"},
        )

    def test_complete_monitoring(self, mock_client, workflows):
        """Test completing workflow monitoring."""
        monitor = LangGraphMultiAgentMonitor(client=mock_client, workflow_name="Test Workflow")
        monitor.start_monitoring()

        outputs = {"result": "completed successfully"}
        summary = monitor.complete_monitoring(outputs=outputs, status="completed")

        workflows.complete.assert_called_once_with(
            workflow_id="workflow_123", status="completed", outputs=outputs
        )
        workflows.analytics.assert_called_once_with("workflow_123")

        assert summary["workflow_id"] == "workflow_123"
        assert summary["status"] == "completed"
        assert summary["outputs"] == outputs
        assert "analytics" in summary
        assert summary["analytics"]["total_cost"] == 0.15


class TestMonitorLangChainAgent:
    """Tests for monitor_langchain_agent helper function."""

    def test_successful_execution(self, mock_client, workflows):
        """Test successful agent execution monitoring."""
        mock_executor = Mock()
        mock_executor.run.return_value = "Agent result"

        result_dict = monitor_langchain_agent(
            client=mock_client,
            agent_executor=mock_executor,
            workflow_name="Test Task",
            agent_name="test_agent",
            inputs={"input": "test query"},
        )

        workflows.create_and_start.assert_called_once_with(name="Test Task", framework="langchain")
        workflows.start.assert_called_once_with("workflow_123", inputs={"input": "test query"})

        # Verify agent was run with callback
        mock_executor.run.assert_called_once()
        assert "callbacks" in mock_executor.run.call_args.kwargs

        # complete() requires a status
        workflows.complete.assert_called_once_with(
            "workflow_123", status="completed", outputs={"result": "Agent result"}
        )

        assert result_dict["result"] == "Agent result"
        assert result_dict["workflow_id"] == "workflow_123"
        assert result_dict["status"] == "completed"

    def test_failed_execution(self, mock_client, workflows):
        """Test failed agent execution monitoring."""
        mock_executor = Mock()
        mock_executor.run.side_effect = RuntimeError("Agent failed")

        result_dict = monitor_langchain_agent(
            client=mock_client,
            agent_executor=mock_executor,
            workflow_name="Test Task",
            agent_name="test_agent",
        )

        complete_args = workflows.complete.call_args
        assert complete_args.kwargs["status"] == "failed"
        assert "error" in complete_args.kwargs["outputs"]

        assert result_dict["result"] is None
        assert result_dict["status"] == "failed"
        assert "Agent failed" in result_dict["error"]


class TestErrorHandling:
    """Tests for error handling."""

    def test_callback_api_error_handling(self, mock_client, workflows, caplog):
        """API errors don't crash the callback, and are logged."""
        workflows.create_execution.side_effect = Exception("API Error")
        workflows.log_interaction.side_effect = Exception("API Error")

        handler = MultiAgentCallbackHandler(
            client=mock_client, workflow_id="workflow_123", agent_name="test"
        )

        with caplog.at_level(logging.WARNING):
            handler.on_chain_start(serialized={}, inputs={})
            handler.on_chain_end(outputs={"result": "test"})

        # Failures are visible in the log, not silently dropped
        assert "Failed to log execution start" in caplog.text
        assert "Failed to log execution end" in caplog.text

        # State should still be reset
        assert handler.execution_start_time is None

    def test_register_agent_before_start(self, mock_client):
        """Test that registering agent before start raises error."""
        monitor = LangGraphMultiAgentMonitor(client=mock_client, workflow_name="Test")

        with pytest.raises(ValueError, match="Must call start_monitoring"):
            monitor.register_agent("test_agent")

    def test_get_callbacks_before_start(self, mock_client):
        """Test that getting callbacks before start raises error."""
        monitor = LangGraphMultiAgentMonitor(client=mock_client, workflow_name="Test")

        with pytest.raises(ValueError, match="Must call start_monitoring"):
            monitor.get_callbacks("test_agent")
