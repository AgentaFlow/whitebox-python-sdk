"""
Unit tests for CrewAI integration.

Tests CrewAI monitoring with mocked API calls.
"""

from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from whiteboxxai.integrations.crewai_monitor import CrewAIMonitor, monitor_crew


@pytest.fixture
def mock_client():
    """Create mock WhiteBoxXAI client."""
    client = Mock()
    client.request = Mock()
    return client


@pytest.fixture
def mock_crew():
    """Create mock CrewAI Crew."""
    # Mock agents
    agent1 = Mock()
    agent1.role = "Research Analyst"
    agent1.goal = "Find accurate information"
    agent1.backstory = "Expert researcher"
    agent1.tools = [Mock(__class__=Mock(__name__="SearchTool"))]
    agent1.llm = Mock(model_name="gpt-4")
    agent1.verbose = True
    agent1.allow_delegation = False
    agent1.max_iter = 15

    agent2 = Mock()
    agent2.role = "Content Writer"
    agent2.goal = "Write engaging content"
    agent2.backstory = "Professional writer"
    agent2.tools = []
    agent2.llm = Mock(model_name="gpt-4")
    agent2.verbose = False
    agent2.allow_delegation = True
    agent2.max_iter = 10

    # Mock tasks
    task1 = Mock()
    task1.description = "Research AI safety regulations"
    task1.expected_output = "List of regulations"
    task1.agent = agent1
    task1.tools = []
    task1.async_execution = False

    task2 = Mock()
    task2.description = "Write article about AI safety"
    task2.expected_output = "1500 word article"
    task2.agent = agent2
    task2.tools = []
    task2.async_execution = False

    # Mock crew
    crew = Mock()
    crew.agents = [agent1, agent2]
    crew.tasks = [task1, task2]
    crew.process = "sequential"

    return crew


class TestCrewAIMonitor:
    """Test CrewAI monitor functionality."""

    def test_init(self, mock_client):
        """Test monitor initialization."""
        with patch(
            "whiteboxxai.integrations.crewai_monitor.WhiteBoxXAI",
            return_value=mock_client,
        ):
            monitor = CrewAIMonitor(api_key="test_api_key")

            assert monitor.client == mock_client
            assert monitor.workflow_id is None
            assert monitor.agent_map == {}
            assert monitor.task_map == {}
            assert monitor.execution_map == {}

    def test_start_monitoring(self, mock_client, mock_crew):
        """Test starting workflow monitoring."""
        with patch(
            "whiteboxxai.integrations.crewai_monitor.WhiteBoxXAI",
            return_value=mock_client,
        ):
            workflow_id = str(uuid4())
            agent1_id = str(uuid4())
            agent2_id = str(uuid4())
            task1_id = str(uuid4())
            task2_id = str(uuid4())

            # Mock API responses
            mock_client.request.side_effect = [
                {"id": workflow_id},  # Create workflow
                {"id": agent1_id},  # Register agent 1
                {"id": agent2_id},  # Register agent 2
                {"id": task1_id},  # Register task 1
                {"id": task2_id},  # Register task 2
                {},  # Start workflow
            ]

            monitor = CrewAIMonitor(api_key="test_api_key")
            result_id = monitor.start_monitoring(
                crew=mock_crew,
                workflow_name="Test Workflow",
                metadata={"project": "test"},
            )

            assert result_id == workflow_id
            assert monitor.workflow_id == workflow_id
            assert len(monitor.agent_map) == 2
            assert len(monitor.task_map) == 2

            # Verify API calls
            calls = mock_client.request.call_args_list

            # Check create workflow call
            assert calls[0][0][0] == "POST"
            assert calls[0][0][1] == "/api/v1/workflows/multi-agent/start"
            assert calls[0][1]["data"]["name"] == "Test Workflow"
            assert calls[0][1]["data"]["framework"] == "crewai"
            assert calls[0][1]["data"]["metadata"]["project"] == "test"

            # Check start workflow call
            assert calls[5][0][0] == "POST"
            assert calls[5][0][1] == f"/api/v1/workflows/multi-agent/{workflow_id}/start"

    def test_register_agent(self, mock_client, mock_crew):
        """Test agent registration."""
        with patch(
            "whiteboxxai.integrations.crewai_monitor.WhiteBoxXAI",
            return_value=mock_client,
        ):
            workflow_id = str(uuid4())
            agent_id = str(uuid4())

            mock_client.request.return_value = {"id": agent_id}

            monitor = CrewAIMonitor(api_key="test_api_key")
            monitor.workflow_id = workflow_id

            result_id = monitor._register_agent(mock_crew.agents[0])

            assert result_id == agent_id
            assert monitor.agent_map[id(mock_crew.agents[0])] == agent_id

            # Verify API call
            call_args = mock_client.request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == f"/api/v1/workflows/multi-agent/{workflow_id}/agents"

            agent_data = call_args[1]["data"]
            assert agent_data["name"] == "Research Analyst"
            assert agent_data["role"] == "Research Analyst"
            assert agent_data["agent_type"] == "crewai_agent"
            assert agent_data["goal"] == "Find accurate information"
            assert agent_data["backstory"] == "Expert researcher"
            assert agent_data["tools"] == ["SearchTool"]
            assert agent_data["llm_provider"] == "gpt"
            assert agent_data["model_name"] == "gpt-4"

    def test_register_task(self, mock_client, mock_crew):
        """Test task registration."""
        with patch(
            "whiteboxxai.integrations.crewai_monitor.WhiteBoxXAI",
            return_value=mock_client,
        ):
            workflow_id = str(uuid4())
            task_id = str(uuid4())
            agent_id = str(uuid4())

            mock_client.request.return_value = {"id": task_id}

            monitor = CrewAIMonitor(api_key="test_api_key")
            monitor.workflow_id = workflow_id
            monitor.agent_map[id(mock_crew.tasks[0].agent)] = agent_id

            result_id = monitor._register_task(mock_crew.tasks[0])

            assert result_id == task_id
            assert monitor.task_map[id(mock_crew.tasks[0])] == task_id

            # Verify API call
            call_args = mock_client.request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == f"/api/v1/workflows/multi-agent/{workflow_id}/tasks"

            task_data = call_args[1]["data"]
            assert task_data["task_name"] == "Research AI safety regulations"
            assert task_data["description"] == "Research AI safety regulations"
            assert task_data["expected_output"] == "List of regulations"
            assert task_data["agent_id"] == agent_id

    def test_log_agent_execution(self, mock_client, mock_crew):
        """Test logging agent execution."""
        with patch(
            "whiteboxxai.integrations.crewai_monitor.WhiteBoxXAI",
            return_value=mock_client,
        ):
            workflow_id = str(uuid4())
            agent_id = str(uuid4())

            mock_client.request.return_value = {}

            monitor = CrewAIMonitor(api_key="test_api_key")
            monitor.workflow_id = workflow_id
            monitor.agent_map[id(mock_crew.agents[0])] = agent_id

            monitor.log_agent_execution(
                agent=mock_crew.agents[0],
                inputs={"query": "test"},
                outputs={"result": "data"},
                tokens_used=100,
                cost=0.002,
            )

            # Verify API call
            call_args = mock_client.request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == f"/api/v1/workflows/multi-agent/{workflow_id}/executions"

            execution_data = call_args[1]["data"]
            assert execution_data["agent_id"] == agent_id
            assert execution_data["inputs"] == {"query": "test"}
            assert execution_data["outputs"] == {"result": "data"}
            assert execution_data["tokens_used"] == 100
            assert execution_data["cost"] == 0.002
            assert execution_data["status"] == "completed"

    def test_log_task_completion(self, mock_client, mock_crew):
        """Test logging task completion."""
        with patch(
            "whiteboxxai.integrations.crewai_monitor.WhiteBoxXAI",
            return_value=mock_client,
        ):
            task_id = str(uuid4())

            mock_client.request.return_value = {}

            monitor = CrewAIMonitor(api_key="test_api_key")
            monitor.task_map[id(mock_crew.tasks[0])] = task_id

            monitor.log_task_completion(
                task=mock_crew.tasks[0],
                status="completed",
                output_data={"result": "success"},
            )

            # Verify API call
            call_args = mock_client.request.call_args
            assert call_args[0][0] == "PATCH"
            assert call_args[0][1] == f"/api/v1/workflows/multi-agent/tasks/{task_id}"

            update_data = call_args[1]["data"]
            assert update_data["status"] == "completed"
            assert update_data["output_data"] == {"result": "success"}

    def test_log_interaction(self, mock_client, mock_crew):
        """Test logging agent interaction."""
        with patch(
            "whiteboxxai.integrations.crewai_monitor.WhiteBoxXAI",
            return_value=mock_client,
        ):
            workflow_id = str(uuid4())
            agent1_id = str(uuid4())
            agent2_id = str(uuid4())

            mock_client.request.return_value = {}

            monitor = CrewAIMonitor(api_key="test_api_key")
            monitor.workflow_id = workflow_id
            monitor.agent_map[id(mock_crew.agents[0])] = agent1_id
            monitor.agent_map[id(mock_crew.agents[1])] = agent2_id

            monitor.log_interaction(
                from_agent=mock_crew.agents[0],
                to_agent=mock_crew.agents[1],
                interaction_type="delegation",
                message="Please write article",
            )

            # Verify API call
            call_args = mock_client.request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == f"/api/v1/workflows/multi-agent/{workflow_id}/interactions"

            interaction_data = call_args[1]["data"]
            assert interaction_data["interaction_type"] == "delegation"
            assert interaction_data["from_agent_id"] == agent1_id
            assert interaction_data["to_agent_id"] == agent2_id
            assert interaction_data["message"] == "Please write article"

    def test_complete_monitoring(self, mock_client):
        """Test completing workflow monitoring."""
        with patch(
            "whiteboxxai.integrations.crewai_monitor.WhiteBoxXAI",
            return_value=mock_client,
        ):
            workflow_id = str(uuid4())

            mock_client.request.side_effect = [
                {},  # Complete workflow
                {"total_tokens": 5000, "total_cost": 0.05},  # Analytics
                {"total_cost": 0.05, "agents": []},  # Cost breakdown
            ]

            monitor = CrewAIMonitor(api_key="test_api_key")
            monitor.workflow_id = workflow_id

            result = monitor.complete_monitoring(status="completed", outputs={"article": "content"})

            assert result["workflow_id"] == workflow_id
            assert result["status"] == "completed"
            assert "analytics" in result

            # Verify complete call
            calls = mock_client.request.call_args_list
            assert calls[0][0][0] == "POST"
            assert calls[0][0][1] == f"/api/v1/workflows/multi-agent/{workflow_id}/complete"

            complete_data = calls[0][1]["data"]
            assert complete_data["status"] == "completed"
            assert complete_data["outputs"] == {"article": "content"}

    def test_get_analytics(self, mock_client):
        """Test getting workflow analytics."""
        with patch(
            "whiteboxxai.integrations.crewai_monitor.WhiteBoxXAI",
            return_value=mock_client,
        ):
            workflow_id = str(uuid4())

            mock_client.request.side_effect = [
                {"total_tokens": 5000, "total_cost": 0.05},
                {"total_cost": 0.05, "agents": []},
            ]

            monitor = CrewAIMonitor(api_key="test_api_key")
            monitor.workflow_id = workflow_id

            analytics = monitor.get_analytics()

            assert "metrics" in analytics
            assert "cost_breakdown" in analytics

            # Verify calls
            calls = mock_client.request.call_args_list
            assert calls[0][0][1] == f"/api/v1/workflows/multi-agent/{workflow_id}/analytics"
            assert calls[1][0][1] == f"/api/v1/workflows/multi-agent/{workflow_id}/cost-breakdown"


class TestMonitorCrewHelper:
    """Test monitor_crew convenience function."""

    def test_monitor_crew(self, mock_client, mock_crew):
        """Test monitor_crew helper function."""
        with patch(
            "whiteboxxai.integrations.crewai_monitor.WhiteBoxXAI",
            return_value=mock_client,
        ):
            workflow_id = str(uuid4())

            mock_client.request.side_effect = [
                {"id": workflow_id},  # Create workflow
                {"id": str(uuid4())},  # Agent 1
                {"id": str(uuid4())},  # Agent 2
                {"id": str(uuid4())},  # Task 1
                {"id": str(uuid4())},  # Task 2
                {},  # Start workflow
            ]

            monitor = monitor_crew(
                crew=mock_crew,
                workflow_name="Test Workflow",
                api_key="test_api_key",
                metadata={"test": "data"},
            )

            assert isinstance(monitor, CrewAIMonitor)
            assert monitor.workflow_id == workflow_id


class TestErrorHandling:
    """Test error handling."""

    def test_start_monitoring_error(self, mock_client, mock_crew):
        """Test error handling in start_monitoring."""
        with patch(
            "whiteboxxai.integrations.crewai_monitor.WhiteBoxXAI",
            return_value=mock_client,
        ):
            mock_client.request.side_effect = Exception("API Error")

            monitor = CrewAIMonitor(api_key="test_api_key")

            with pytest.raises(Exception, match="API Error"):
                monitor.start_monitoring(crew=mock_crew, workflow_name="Test Workflow")

    def test_log_execution_unregistered_agent(self, mock_client, mock_crew):
        """Test logging execution for unregistered agent."""
        with patch(
            "whiteboxxai.integrations.crewai_monitor.WhiteBoxXAI",
            return_value=mock_client,
        ):
            monitor = CrewAIMonitor(api_key="test_api_key")
            monitor.workflow_id = str(uuid4())

            # Should not raise exception, just log warning
            monitor.log_agent_execution(agent=mock_crew.agents[0], inputs={"test": "data"})

            # No API call should be made
            mock_client.request.assert_not_called()

    def test_complete_monitoring_no_workflow(self, mock_client):
        """Test completing monitoring without active workflow."""
        with patch(
            "whiteboxxai.integrations.crewai_monitor.WhiteBoxXAI",
            return_value=mock_client,
        ):
            monitor = CrewAIMonitor(api_key="test_api_key")

            with pytest.raises(Exception):
                monitor.complete_monitoring(status="completed")
