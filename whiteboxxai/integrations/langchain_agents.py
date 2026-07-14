"""
LangChain Multi-Agent Integration for WhiteBoxXAI

Enhanced callback handler for monitoring multi-agent LangChain workflows including:
- LangGraph multi-agent patterns
- Agent supervisors and coordinators
- Tool usage and agent handoffs
- Agent-to-agent communication
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.schema.document import Document
from langchain.schema.output import ChatGeneration, Generation

try:
    from whiteboxxai import WhiteBoxXAI
except ImportError:
    WhiteBoxXAI = None


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
        workflow_id = client.agent_workflows.create(
            name="Research Workflow",
            framework="langchain"
        ).id

        # Start workflow
        client.agent_workflows.start(workflow_id)

        # Create callback
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
            outputs={"result": result}
        )
        ```
    """

    def __init__(
        self,
        client: "WhiteBoxXAI",
        workflow_id: str,
        agent_name: str = "main",
        agent_role: Optional[str] = None,
        track_tokens: bool = True,
        track_costs: bool = True,
    ):
        """Initialize the callback handler.

        Args:
            client: WhiteBoxXAI client instance
            workflow_id: ID of the workflow to track
            agent_name: Name of the current agent
            agent_role: Role/description of the agent
            track_tokens: Whether to track token usage
            track_costs: Whether to estimate costs
        """
        if WhiteBoxXAI is None:
            raise ImportError(
                "whiteboxxai package not installed. " "Install with: pip install whiteboxxai"
            )

        self.client = client
        self.workflow_id = workflow_id
        self.agent_name = agent_name
        self.agent_role = agent_role or agent_name
        self.track_tokens = track_tokens
        self.track_costs = track_costs

        # Tracking state
        self.current_execution_id: Optional[str] = None
        self.execution_start_time: Optional[datetime] = None
        self.llm_call_count = 0
        self.tool_call_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.execution_inputs: Optional[Dict[str, Any]] = None

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts."""
        # Start agent execution
        self.execution_start_time = datetime.utcnow()
        self.execution_inputs = inputs
        self.llm_call_count = 0
        self.tool_call_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends successfully."""
        if self.execution_start_time:
            duration_ms = int(
                (datetime.utcnow() - self.execution_start_time).total_seconds() * 1000
            )

            # Log agent execution
            try:
                response = self.client.agent_workflows.create_execution(
                    workflow_id=self.workflow_id,
                    agent_name=self.agent_name,
                    status="completed",
                    inputs=self.execution_inputs,
                    outputs=outputs,
                    duration_ms=duration_ms,
                    llm_call_count=self.llm_call_count,
                    tool_call_count=self.tool_call_count,
                    tokens_used=self.total_tokens if self.track_tokens else None,
                    cost=self.total_cost if self.track_costs else None,
                )
                self.current_execution_id = response.get("id")
            except Exception as e:
                print(f"Warning: Failed to log execution: {e}")

            # Reset state
            self.execution_start_time = None

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Run when chain errors."""
        if self.execution_start_time:
            duration_ms = int(
                (datetime.utcnow() - self.execution_start_time).total_seconds() * 1000
            )

            # Log failed execution
            try:
                self.client.agent_workflows.create_execution(
                    workflow_id=self.workflow_id,
                    agent_name=self.agent_name,
                    status="failed",
                    inputs=self.execution_inputs,
                    outputs={"error": str(error)},
                    duration_ms=duration_ms,
                    llm_call_count=self.llm_call_count,
                    tool_call_count=self.tool_call_count,
                )
            except Exception as e:
                print(f"Warning: Failed to log error: {e}")

            self.execution_start_time = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
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

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """Run when agent takes an action (tool call)."""
        self.tool_call_count += 1

        # Log tool call as interaction
        try:
            self.client.agent_workflows.create_interaction(
                workflow_id=self.workflow_id,
                from_agent=self.agent_name,
                to_agent="tool",
                interaction_type="tool_call",
                message=f"Tool: {action.tool}, Input: {action.tool_input}",
                meta_data={
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                    "log": action.log,
                },
            )
        except Exception as e:
            print(f"Warning: Failed to log tool call: {e}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run when agent finishes execution."""
        # This is called when the agent completes its reasoning
        pass

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Run when tool starts."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends."""
        # Log tool result as interaction
        try:
            self.client.agent_workflows.create_interaction(
                workflow_id=self.workflow_id,
                from_agent="tool",
                to_agent=self.agent_name,
                interaction_type="response",
                message=f"Tool result: {output[:500]}",  # Truncate long outputs
                meta_data={"output": output},
            )
        except Exception as e:
            print(f"Warning: Failed to log tool result: {e}")

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Run when tool errors."""
        try:
            self.client.agent_workflows.create_interaction(
                workflow_id=self.workflow_id,
                from_agent="tool",
                to_agent=self.agent_name,
                interaction_type="response",
                message=f"Tool error: {str(error)}",
                meta_data={"error": str(error), "error_type": type(error).__name__},
            )
        except Exception as e:
            print(f"Warning: Failed to log tool error: {e}")

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""
        pass


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
        meta_data: Optional[Dict[str, Any]] = None,
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
        self.workflow_id: Optional[str] = None
        self.callbacks: Dict[str, MultiAgentCallbackHandler] = {}
        self.start_time: Optional[datetime] = None

    def start_monitoring(self, inputs: Optional[Dict[str, Any]] = None) -> str:
        """Start workflow monitoring.

        Args:
            inputs: Initial workflow inputs

        Returns:
            workflow_id: ID of the created workflow
        """
        self.start_time = datetime.utcnow()

        # Create workflow
        response = self.client.agent_workflows.create(
            name=self.workflow_name,
            framework="langchain",
            inputs=inputs,
            meta_data=self.workflow_meta_data,
        )
        self.workflow_id = response.get("id")

        # Start workflow
        self.client.agent_workflows.start(self.workflow_id)

        return self.workflow_id

    def register_agent(
        self,
        agent_name: str,
        role: Optional[str] = None,
        model_name: Optional[str] = None,
        tools: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Register an agent in the workflow.

        Args:
            agent_name: Name of the agent
            role: Agent's role/goal
            model_name: LLM model used
            tools: List of tool names
            **kwargs: Additional agent configuration
        """
        if not self.workflow_id:
            raise ValueError("Must call start_monitoring() first")

        self.client.agent_workflows.register_agent(
            workflow_id=self.workflow_id,
            name=agent_name,
            role=role or agent_name,
            model_name=model_name,
            tools=tools,
            **kwargs,
        )

    def get_callbacks(
        self, agent_name: str, agent_role: Optional[str] = None
    ) -> List[BaseCallbackHandler]:
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
            )

        return [self.callbacks[agent_name]]

    def log_handoff(
        self,
        from_agent: str,
        to_agent: str,
        message: str,
        meta_data: Optional[Dict[str, Any]] = None,
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

        self.client.agent_workflows.create_interaction(
            workflow_id=self.workflow_id,
            from_agent=from_agent,
            to_agent=to_agent,
            interaction_type="handoff",
            message=message,
            meta_data=meta_data,
        )

    def complete_monitoring(
        self, outputs: Optional[Dict[str, Any]] = None, status: str = "completed"
    ) -> Dict[str, Any]:
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
            workflow_id=self.workflow_id, outputs=outputs, status=status
        )

        # Get analytics
        try:
            analytics = self.client.agent_workflows.get_analytics(self.workflow_id)
            return {
                "workflow_id": self.workflow_id,
                "status": status,
                "outputs": outputs,
                "analytics": analytics,
            }
        except Exception as e:
            print(f"Warning: Failed to retrieve analytics: {e}")
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
    inputs: Optional[Dict[str, Any]] = None,
    **run_kwargs,
) -> Dict[str, Any]:
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
    response = client.agent_workflows.create(
        name=workflow_name, framework="langchain", inputs=inputs
    )
    workflow_id = response.get("id")

    # Start workflow
    client.agent_workflows.start(workflow_id)

    # Create callback
    callback = MultiAgentCallbackHandler(
        client=client, workflow_id=workflow_id, agent_name=agent_name
    )

    try:
        # Run agent with callback
        result = agent_executor.run(callbacks=[callback], **run_kwargs)

        # Complete workflow
        client.agent_workflows.complete(workflow_id, outputs={"result": result})

        return {"result": result, "workflow_id": workflow_id, "status": "completed"}
    except Exception as e:
        # Log failure
        client.agent_workflows.complete(workflow_id, outputs={"error": str(e)}, status="failed")

        return {
            "result": None,
            "workflow_id": workflow_id,
            "status": "failed",
            "error": str(e),
        }
