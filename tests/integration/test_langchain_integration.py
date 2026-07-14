"""Unit tests for LangChain multi-agent integration."""

from datetime import datetime
from unittest.mock import MagicMock, Mock, call, patch

import pytest

# NOTE: this requires the real `langchain` package (see the `langchain` extra
# in sdk/pyproject.toml) to be installed. langchain_agents.py subclasses
# langchain's real BaseCallbackHandler at class-definition time, so faking
# out sys.modules["langchain.callbacks.base"] etc. here (as this file
# previously did) makes that inheritance operate on a Mock instead of a
# real class, producing undefined/garbage behavior.
from whiteboxxai.integrations.langchain_agents import (
    LangGraphMultiAgentMonitor,
    MultiAgentCallbackHandler,
    monitor_langchain_agent,
)


@pytest.fixture
def mock_client():
    """Create a mock WhiteBoxXAI client."""
    client = Mock()
    client.agent_workflows = Mock()
    return client


@pytest.fixture
def mock_workflow_response():
    """Mock workflow creation response."""
    return {"id": "workflow_123", "status": "pending"}


@pytest.fixture
def callback_handler(mock_client, mock_workflow_response):
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
        assert handler.track_tokens is True
        assert handler.track_costs is True
        assert handler.llm_call_count == 0
        assert handler.tool_call_count == 0

    def test_on_chain_start(self, callback_handler):
        """Test chain start callback."""
        inputs = {"input": "test query"}

        callback_handler.on_chain_start(serialized={"name": "test_chain"}, inputs=inputs)

        assert callback_handler.execution_inputs == inputs
        assert callback_handler.execution_start_time is not None
        assert callback_handler.llm_call_count == 0
        assert callback_handler.tool_call_count == 0

    def test_on_chain_end(self, callback_handler, mock_client):
        """Test chain end callback."""
        callback_handler.execution_start_time = datetime.utcnow()
        callback_handler.execution_inputs = {"input": "test"}
        callback_handler.llm_call_count = 2
        callback_handler.tool_call_count = 1
        callback_handler.total_tokens = 150
        callback_handler.total_cost = 0.003

        mock_client.agent_workflows.create_execution.return_value = {"id": "exec_123"}

        outputs = {"output": "test result"}
        callback_handler.on_chain_end(outputs=outputs)

        # Verify execution was logged
        mock_client.agent_workflows.create_execution.assert_called_once()
        call_args = mock_client.agent_workflows.create_execution.call_args

        assert call_args.kwargs["workflow_id"] == "workflow_123"
        assert call_args.kwargs["agent_name"] == "test_agent"
        assert call_args.kwargs["status"] == "completed"
        assert call_args.kwargs["inputs"] == {"input": "test"}
        assert call_args.kwargs["outputs"] == outputs
        assert call_args.kwargs["llm_call_count"] == 2
        assert call_args.kwargs["tool_call_count"] == 1
        assert call_args.kwargs["tokens_used"] == 150
        assert call_args.kwargs["cost"] == 0.003

        # Verify state was reset
        assert callback_handler.execution_start_time is None

    def test_on_chain_error(self, callback_handler, mock_client):
        """Test chain error callback."""
        callback_handler.execution_start_time = datetime.utcnow()
        callback_handler.execution_inputs = {"input": "test"}
        callback_handler.llm_call_count = 1

        error = ValueError("Test error")
        callback_handler.on_chain_error(error=error)

        # Verify failed execution was logged
        mock_client.agent_workflows.create_execution.assert_called_once()
        call_args = mock_client.agent_workflows.create_execution.call_args

        assert call_args.kwargs["status"] == "failed"
        assert "error" in call_args.kwargs["outputs"]
        assert "Test error" in call_args.kwargs["outputs"]["error"]

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

    def test_on_agent_action(self, callback_handler, mock_client):
        """Test agent action (tool call) callback."""
        # Mock AgentAction
        mock_action = Mock()
        mock_action.tool = "search"
        mock_action.tool_input = "AI safety"
        mock_action.log = "Searching for AI safety"

        callback_handler.on_agent_action(action=mock_action)

        # Verify tool call was logged as interaction
        mock_client.agent_workflows.create_interaction.assert_called_once()
        call_args = mock_client.agent_workflows.create_interaction.call_args

        assert call_args.kwargs["workflow_id"] == "workflow_123"
        assert call_args.kwargs["from_agent"] == "test_agent"
        assert call_args.kwargs["to_agent"] == "tool"
        assert call_args.kwargs["interaction_type"] == "tool_call"
        assert "search" in call_args.kwargs["message"]

        # Verify tool count incremented
        assert callback_handler.tool_call_count == 1

    def test_on_tool_end(self, callback_handler, mock_client):
        """Test tool end callback."""
        output = "Search results for AI safety"

        callback_handler.on_tool_end(output=output)

        # Verify tool result was logged as interaction
        mock_client.agent_workflows.create_interaction.assert_called_once()
        call_args = mock_client.agent_workflows.create_interaction.call_args

        assert call_args.kwargs["from_agent"] == "tool"
        assert call_args.kwargs["to_agent"] == "test_agent"
        assert call_args.kwargs["interaction_type"] == "response"
        assert output in call_args.kwargs["message"]

    def test_on_tool_error(self, callback_handler, mock_client):
        """Test tool error callback."""
        error = RuntimeError("Tool failed")

        callback_handler.on_tool_error(error=error)

        # Verify tool error was logged
        mock_client.agent_workflows.create_interaction.assert_called_once()
        call_args = mock_client.agent_workflows.create_interaction.call_args

        assert call_args.kwargs["interaction_type"] == "response"
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

    def test_start_monitoring(self, mock_client, mock_workflow_response):
        """Test starting workflow monitoring."""
        mock_client.agent_workflows.create.return_value = mock_workflow_response

        monitor = LangGraphMultiAgentMonitor(client=mock_client, workflow_name="Test Workflow")

        inputs = {"query": "test"}
        workflow_id = monitor.start_monitoring(inputs=inputs)

        # Verify workflow was created
        mock_client.agent_workflows.create.assert_called_once_with(
            name="Test Workflow", framework="langchain", inputs=inputs, meta_data={}
        )

        # Verify workflow was started
        mock_client.agent_workflows.start.assert_called_once_with("workflow_123")

        assert workflow_id == "workflow_123"
        assert monitor.workflow_id == "workflow_123"
        assert monitor.start_time is not None

    def test_register_agent(self, mock_client, mock_workflow_response):
        """Test agent registration."""
        mock_client.agent_workflows.create.return_value = mock_workflow_response

        monitor = LangGraphMultiAgentMonitor(client=mock_client, workflow_name="Test Workflow")
        monitor.start_monitoring()

        monitor.register_agent(
            agent_name="researcher",
            role="Research Agent",
            model_name="gpt-4",
            tools=["search", "scrape"],
        )

        # Verify agent was registered
        mock_client.agent_workflows.register_agent.assert_called_once_with(
            workflow_id="workflow_123",
            name="researcher",
            role="Research Agent",
            model_name="gpt-4",
            tools=["search", "scrape"],
        )

    def test_get_callbacks(self, mock_client, mock_workflow_response):
        """Test getting callbacks for an agent."""
        mock_client.agent_workflows.create.return_value = mock_workflow_response

        monitor = LangGraphMultiAgentMonitor(client=mock_client, workflow_name="Test Workflow")
        monitor.start_monitoring()

        callbacks = monitor.get_callbacks("researcher", agent_role="Research Agent")

        assert len(callbacks) == 1
        assert isinstance(callbacks[0], MultiAgentCallbackHandler)
        assert callbacks[0].agent_name == "researcher"
        assert callbacks[0].agent_role == "Research Agent"

        # Getting callbacks again should return the same instance
        callbacks2 = monitor.get_callbacks("researcher")
        assert callbacks2[0] is callbacks[0]

    def test_log_handoff(self, mock_client, mock_workflow_response):
        """Test logging agent-to-agent handoff."""
        mock_client.agent_workflows.create.return_value = mock_workflow_response

        monitor = LangGraphMultiAgentMonitor(client=mock_client, workflow_name="Test Workflow")
        monitor.start_monitoring()

        monitor.log_handoff(
            from_agent="supervisor",
            to_agent="researcher",
            message="Please research this topic",
            meta_data={"topic": "AI safety"},
        )

        # Verify handoff was logged
        mock_client.agent_workflows.create_interaction.assert_called_once_with(
            workflow_id="workflow_123",
            from_agent="supervisor",
            to_agent="researcher",
            interaction_type="handoff",
            message="Please research this topic",
            meta_data={"topic": "AI safety"},
        )

    def test_complete_monitoring(self, mock_client, mock_workflow_response):
        """Test completing workflow monitoring."""
        mock_client.agent_workflows.create.return_value = mock_workflow_response
        mock_client.agent_workflows.get_analytics.return_value = {
            "total_cost": 0.15,
            "total_tokens": 750,
        }

        monitor = LangGraphMultiAgentMonitor(client=mock_client, workflow_name="Test Workflow")
        monitor.start_monitoring()

        outputs = {"result": "completed successfully"}
        summary = monitor.complete_monitoring(outputs=outputs, status="completed")

        # Verify workflow was completed
        mock_client.agent_workflows.complete.assert_called_once_with(
            workflow_id="workflow_123", outputs=outputs, status="completed"
        )

        # Verify analytics were retrieved
        mock_client.agent_workflows.get_analytics.assert_called_once_with("workflow_123")

        assert summary["workflow_id"] == "workflow_123"
        assert summary["status"] == "completed"
        assert summary["outputs"] == outputs
        assert "analytics" in summary
        assert summary["analytics"]["total_cost"] == 0.15


class TestMonitorLangChainAgent:
    """Tests for monitor_langchain_agent helper function."""

    def test_successful_execution(self, mock_client, mock_workflow_response):
        """Test successful agent execution monitoring."""
        mock_client.agent_workflows.create.return_value = mock_workflow_response

        # Mock agent executor
        mock_executor = Mock()
        mock_executor.run.return_value = "Agent result"

        result_dict = monitor_langchain_agent(
            client=mock_client,
            agent_executor=mock_executor,
            workflow_name="Test Task",
            agent_name="test_agent",
            inputs={"input": "test query"},
        )

        # Verify workflow was created
        mock_client.agent_workflows.create.assert_called_once()

        # Verify workflow was started
        mock_client.agent_workflows.start.assert_called_once_with("workflow_123")

        # Verify agent was run with callback
        mock_executor.run.assert_called_once()
        assert "callbacks" in mock_executor.run.call_args.kwargs

        # Verify workflow was completed
        mock_client.agent_workflows.complete.assert_called_once()
        complete_args = mock_client.agent_workflows.complete.call_args
        assert complete_args.args[0] == "workflow_123"
        assert complete_args.kwargs["outputs"] == {"result": "Agent result"}

        # Verify result
        assert result_dict["result"] == "Agent result"
        assert result_dict["workflow_id"] == "workflow_123"
        assert result_dict["status"] == "completed"

    def test_failed_execution(self, mock_client, mock_workflow_response):
        """Test failed agent execution monitoring."""
        mock_client.agent_workflows.create.return_value = mock_workflow_response

        # Mock agent executor that raises error
        mock_executor = Mock()
        mock_executor.run.side_effect = RuntimeError("Agent failed")

        result_dict = monitor_langchain_agent(
            client=mock_client,
            agent_executor=mock_executor,
            workflow_name="Test Task",
            agent_name="test_agent",
        )

        # Verify workflow was marked as failed
        complete_args = mock_client.agent_workflows.complete.call_args
        assert complete_args.kwargs["status"] == "failed"
        assert "error" in complete_args.kwargs["outputs"]

        # Verify result
        assert result_dict["result"] is None
        assert result_dict["status"] == "failed"
        assert "Agent failed" in result_dict["error"]


class TestErrorHandling:
    """Tests for error handling."""

    def test_callback_api_error_handling(self, mock_client):
        """Test that API errors don't crash the callback."""
        mock_client.agent_workflows.create_execution.side_effect = Exception("API Error")

        handler = MultiAgentCallbackHandler(
            client=mock_client, workflow_id="workflow_123", agent_name="test"
        )

        handler.execution_start_time = datetime.utcnow()
        handler.execution_inputs = {}

        # Should not raise exception
        handler.on_chain_end(outputs={"result": "test"})

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
