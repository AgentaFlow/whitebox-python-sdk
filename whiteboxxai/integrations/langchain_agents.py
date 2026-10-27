"""
LangChain Multi-Agent Integration for WhiteBoxXAI

Enhanced callback handler for monitoring multi-agent LangChain workflows including:
- LangGraph multi-agent patterns
- Agent supervisors and coordinators
- Tool usage and agent handoffs
- Agent-to-agent communication
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

# langchain_core is the modern, lighter-weight home for these types; fall
# back to the legacy langchain.schema.* locations for older installs (mirrors
# the version-tolerant pattern in integrations/langchain.py).
try:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError:
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import AgentAction, AgentFinish, LLMResult

try:
    from whiteboxxai import WhiteBoxXAI
except ImportError:
    WhiteBoxXAI = None

logger = logging.getLogger(__name__)


class MultiAgentCallbackHandler(BaseCallbackHandler):
    """Enhanced callback handler for multi-agent LangChain workflows.

    This handler tracks:
    - Agent executions and decisions
    - Tool calls and results
    - Agent-to-agent handoffs
    - LLM calls per agent
    - Workflow-level metrics

    Example:
        ```python
        from langchain.agents import AgentExecutor, create_react_agent
        from whiteboxxai.integrations import MultiAgentCallbackHandler

        # Initialize WhiteBoxXAI client
        client = WhiteBoxXAI(api_key="your_key")

        # Create workflow
        workflow_id = client.agent_workflows.create_and_start(
            name="Research Workflow",
            framework="langchain"
        )["id"]

        # Start workflow
        client.agent_workflows.start(workflow_id)

        # Create callback (the "researcher" agent is registered on first use;
        # pass agent_id=... to reuse one from register_agent())
        callback = MultiAgentCallbackHandler(
            client=client,
            workflow_id=workflow_id,
            agent_name="researcher"
        )

        # Use with agent
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            callbacks=[callback]
        )
        result = agent_executor.run("Research AI safety")

        # Complete workflow
        client.agent_workflows.complete(
            workflow_id,
            status="completed",
            outputs={"result": result}
        )
        ```

    Note:
        The API has no route to close an agent execution, so the handler opens
        one per top-level chain run (``create_execution``) and logs how it ended
        -- duration, LLM/tool call counts, tokens, cost, outputs or error -- as
        an ``execution_end`` interaction on that execution.
    """

    def __init__(
        self,
        client: "WhiteBoxXAI",
        workflow_id: str,
        agent_name: str = "main",
        agent_role: str | None = None,
        track_tokens: bool = True,
        track_costs: bool = True,
        agent_id: str | None = None,
    ):
        """Initialize the callback handler.

        Args:
            client: WhiteBoxXAI client instance
            workflow_id: ID of the workflow to track
            agent_name: Name of the current agent
            agent_role: Role/description of the agent
            track_tokens: Whether to track token usage
            track_costs: Whether to estimate costs
            agent_id: ID of this agent as returned by
                ``client.agent_workflows.register_agent()``. If omitted, the agent
                is registered under ``agent_name`` the first time it is needed.
        """
        if WhiteBoxXAI is None:
            raise ImportError(
                "whiteboxxai package not installed. " "Install with: pip install whiteboxxai"
            )

        self.client = client
        self.workflow_id = workflow_id
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_role = agent_role or agent_name
        self.track_tokens = track_tokens
        self.track_costs = track_costs

        # Tracking state
        self.current_execution_id: str | None = None
        self.execution_start_time: datetime | None = None
        self.llm_call_count = 0
        self.tool_call_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.execution_inputs: dict[str, Any] | None = None

    def _ensure_agent_id(self) -> str:
        """Return this agent's ID, registering it in the workflow on first use."""
        if self.agent_id is None:
            agent = self.client.agent_workflows.register_agent(
                self.workflow_id, name=self.agent_name, role=self.agent_role
            )
            self.agent_id = agent["id"]
        return self.agent_id

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts."""
        if kwargs.get("parent_run_id") is not None:
            return  # a nested chain; the top-level run is the agent execution

        self.execution_start_time = datetime.utcnow()
        self.execution_inputs = inputs
        self.llm_call_count = 0
        self.tool_call_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0

        # Open the agent execution
        try:
            execution = self.client.agent_workflows.create_execution(
                self.workflow_id, self._ensure_agent_id(), inputs=inputs
            )
            self.current_execution_id = execution.get("id")
        except Exception as e:
            logger.warning("Failed to log execution start: %s", e)

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends successfully."""
        if kwargs.get("parent_run_id") is None:
            self._log_execution_end("completed", {"outputs": outputs})

    def on_chain_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> None:
        """Run when chain errors."""
        if kwargs.get("parent_run_id") is None:
            self._log_execution_end("failed", {"error": str(error)})

    def _log_execution_end(self, status: str, result: dict[str, Any]) -> None:
        """Log how the current execution ended, then reset the run state."""
        if not self.execution_start_time:
            return

        duration_ms = int((datetime.utcnow() - self.execution_start_time).total_seconds() * 1000)
        try:
            self.client.agent_workflows.log_interaction(
                self.workflow_id,
                interaction_type="execution_end",
                from_agent_id=self._ensure_agent_id(),
                from_execution_id=self.current_execution_id,
                message=status,
                meta_data={
                    "status": status,
                    "duration_ms": duration_ms,
                    "llm_call_count": self.llm_call_count,
                    "tool_call_count": self.tool_call_count,
                    "tokens_used": self.total_tokens if self.track_tokens else None,
                    "cost": self.total_cost if self.track_costs else None,
                    **result,
                },
            )
        except Exception as e:
            logger.warning("Failed to log execution end: %s", e)

        self.execution_start_time = None
        self.current_execution_id = None

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        """Run when LLM starts."""
        self.llm_call_count += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends."""
        # Track tokens if available
        if self.track_tokens and hasattr(response, "llm_output"):
            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage", {})

            total = token_usage.get("total_tokens", 0)
            self.total_tokens += total

            # Estimate cost if tracking
            if self.track_costs and total > 0:
                # Rough estimate: $0.002 per 1K tokens (GPT-3.5 pricing)
                self.total_cost += (total / 1000) * 0.002

    def _log_tool_interaction(
        self, interaction_type: str, message: str, meta_data: dict[str, Any]
    ) -> None:
        """Log a tool call/result as an interaction from this agent."""
        try:
            self.client.agent_workflows.log_interaction(
                self.workflow_id,
                interaction_type=interaction_type,
                from_agent_id=self._ensure_agent_id(),
                from_execution_id=self.current_execution_id,
                message=message,
                meta_data=meta_data,
            )
        except Exception as e:
            logger.warning("Failed to log %s: %s", interaction_type, e)

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """Run when agent takes an action (tool call)."""
        self.tool_call_count += 1
        self._log_tool_interaction(
            "tool_call",
            f"Tool: {action.tool}, Input: {action.tool_input}",
            {
                "tool": action.tool,
                "tool_input": action.tool_input,
                "log": action.log,
            },
        )

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run when agent finishes execution."""
        # This is called when the agent completes its reasoning

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Run when tool starts."""

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends."""
        self._log_tool_interaction(
            "tool_result",
            f"Tool result: {output[:500]}",  # Truncate long outputs
            {"output": output},
        )

    def on_tool_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> None:
        """Run when tool errors."""
        self._log_tool_interaction(
            "tool_result",
            f"Tool error: {str(error)}",
            {"error": str(error), "error_type": type(error).__name__},
        )

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""


class LangGraphMultiAgentMonitor:
    """Monitor for LangGraph multi-agent workflows.

    Provides higher-level monitoring for LangGraph patterns like:
    - Agent supervisors
    - Agent networks
    - Sequential/parallel agent execution

    Example:
        ```python
        from langgraph.graph import StateGraph
        from whiteboxxai.integrations import LangGraphMultiAgentMonitor

        # Create monitor
        monitor = LangGraphMultiAgentMonitor(
            client=client,
            workflow_name="Multi-Agent Research"
        )

        # Start monitoring
        workflow_id = monitor.start_monitoring()

        # Register agents
        monitor.register_agent("supervisor", role="Coordinates other agents")
        monitor.register_agent("researcher", role="Gathers information")
        monitor.register_agent("writer", role="Writes content")

        # Execute graph with callbacks
        graph = StateGraph(...)
        result = graph.invoke(
            inputs,
            config={"callbacks": [monitor.get_callbacks("supervisor")]}
        )

        # Complete monitoring
        monitor.complete_monitoring(outputs={"result": result})
        ```
    """

    def __init__(
        self,
        client: "WhiteBoxXAI",
        workflow_name: str,
        meta_data: dict[str, Any] | None = None,
    ):
        """Initialize the LangGraph monitor.

        Args:
            client: WhiteBoxXAI client instance
            workflow_name: Name for the workflow
            meta_data: Additional meta_data to attach
        """
        if WhiteBoxXAI is None:
            raise ImportError(
                "whiteboxxai package not installed. " "Install with: pip install whiteboxxai"
            )

        self.client = client
        self.workflow_name = workflow_name
        self.workflow_meta_data = meta_data or {}
        self.workflow_id: str | None = None
        self.callbacks: dict[str, MultiAgentCallbackHandler] = {}
        self.agent_ids: dict[str, str] = {}
        self.start_time: datetime | None = None

    def start_monitoring(self, inputs: dict[str, Any] | None = None) -> str:
        """Start workflow monitoring.

        Args:
            inputs: Initial workflow inputs

        Returns:
            workflow_id: ID of the created workflow
        """
        self.start_time = datetime.utcnow()

        # Create workflow, then mark it running with its inputs
        response = self.client.agent_workflows.create_and_start(
            name=self.workflow_name,
            framework="langchain",
            meta_data=self.workflow_meta_data,
        )
        self.workflow_id = response.get("id")
        self.client.agent_workflows.start(self.workflow_id, inputs=inputs)

        return self.workflow_id

    def register_agent(
        self,
        agent_name: str,
        role: str | None = None,
        model_name: str | None = None,
        tools: list[str] | None = None,
        **kwargs,
    ) -> str:
        """Register an agent in the workflow.

        Args:
            agent_name: Name of the agent
            role: Agent's role/goal
            model_name: LLM model used
            tools: List of tool names
            **kwargs: Additional agent configuration

        Returns:
            agent_id: ID of the registered agent
        """
        if not self.workflow_id:
            raise ValueError("Must call start_monitoring() first")

        agent = self.client.agent_workflows.register_agent(
            workflow_id=self.workflow_id,
            name=agent_name,
            role=role or agent_name,
            model_name=model_name,
            tools=tools,
            **kwargs,
        )
        self.agent_ids[agent_name] = agent["id"]
        return agent["id"]

    def get_callbacks(
        self, agent_name: str, agent_role: str | None = None
    ) -> list[BaseCallbackHandler]:
        """Get callbacks for a specific agent.

        Args:
            agent_name: Name of the agent
            agent_role: Optional role description

        Returns:
            List of callback handlers
        """
        if not self.workflow_id:
            raise ValueError("Must call start_monitoring() first")

        if agent_name not in self.callbacks:
            self.callbacks[agent_name] = MultiAgentCallbackHandler(
                client=self.client,
                workflow_id=self.workflow_id,
                agent_name=agent_name,
                agent_role=agent_role,
                agent_id=self.agent_ids.get(agent_name),
            )

        return [self.callbacks[agent_name]]

    def log_handoff(
        self,
        from_agent: str,
        to_agent: str,
        message: str,
        meta_data: dict[str, Any] | None = None,
    ) -> None:
        """Log an agent-to-agent handoff.

        Args:
            from_agent: Agent passing control
            to_agent: Agent receiving control
            message: Handoff message/context
            meta_data: Additional meta_data
        """
        if not self.workflow_id:
            raise ValueError("Must call start_monitoring() first")

        self.client.agent_workflows.log_interaction(
            workflow_id=self.workflow_id,
            interaction_type="handoff",
            from_agent_id=self._agent_id(from_agent),
            to_agent_id=self._agent_id(to_agent),
            message=message,
            meta_data=meta_data,
        )

    def _agent_id(self, agent_name: str) -> str:
        """ID of a registered agent, registering it first if it isn't yet."""
        if agent_name not in self.agent_ids:
            self.register_agent(agent_name)
        return self.agent_ids[agent_name]

    def complete_monitoring(
        self, outputs: dict[str, Any] | None = None, status: str = "completed"
    ) -> dict[str, Any]:
        """Complete workflow monitoring.

        Args:
            outputs: Final workflow outputs
            status: Workflow status (completed/failed)

        Returns:
            Summary with analytics
        """
        if not self.workflow_id:
            raise ValueError("Must call start_monitoring() first")

        # Complete workflow
        self.client.agent_workflows.complete(
            workflow_id=self.workflow_id, status=status, outputs=outputs
        )

        # Get analytics
        try:
            analytics = self.client.agent_workflows.analytics(self.workflow_id)
            return {
                "workflow_id": self.workflow_id,
                "status": status,
                "outputs": outputs,
                "analytics": analytics,
            }
        except Exception as e:
            logger.warning("Failed to retrieve analytics: %s", e)
            return {
                "workflow_id": self.workflow_id,
                "status": status,
                "outputs": outputs,
            }


def monitor_langchain_agent(
    client: "WhiteBoxXAI",
    agent_executor: Any,
    workflow_name: str,
    agent_name: str = "main",
    inputs: dict[str, Any] | None = None,
    **run_kwargs,
) -> dict[str, Any]:
    """Helper function to monitor a single LangChain agent execution.

    Args:
        client: WhiteBoxXAI client
        agent_executor: LangChain AgentExecutor instance
        workflow_name: Name for the workflow
        agent_name: Name of the agent
        inputs: Inputs to the agent
        **run_kwargs: Additional arguments to pass to agent.run()

    Returns:
        Dict with result and workflow_id

    Example:
        ```python
        from langchain.agents import AgentExecutor, create_react_agent
        from whiteboxxai.integrations import monitor_langchain_agent

        result_dict = monitor_langchain_agent(
            client=client,
            agent_executor=agent_executor,
            workflow_name="Research Task",
            agent_name="researcher",
            inputs={"input": "Research AI safety"}
        )

        print(f"Result: {result_dict['result']}")
        print(f"Workflow ID: {result_dict['workflow_id']}")
        ```
    """
    # Create workflow
    response = client.agent_workflows.create_and_start(name=workflow_name, framework="langchain")
    workflow_id = response.get("id")

    # Start workflow
    client.agent_workflows.start(workflow_id, inputs=inputs)

    # Create callback
    callback = MultiAgentCallbackHandler(
        client=client, workflow_id=workflow_id, agent_name=agent_name
    )

    try:
        # Run agent with callback
        result = agent_executor.run(callbacks=[callback], **run_kwargs)

        # Complete workflow
        client.agent_workflows.complete(workflow_id, status="completed", outputs={"result": result})

        return {"result": result, "workflow_id": workflow_id, "status": "completed"}
    except Exception as e:
        # Log failure
        client.agent_workflows.complete(workflow_id, status="failed", outputs={"error": str(e)})

        return {
            "result": None,
            "workflow_id": workflow_id,
            "status": "failed",
            "error": str(e),
        }
