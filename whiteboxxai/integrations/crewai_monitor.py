"""
CrewAI Integration for WhiteBoxXAI

Monitor CrewAI multi-agent workflows with automatic tracking of agents,
tasks, interactions, and costs.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from whiteboxxai import WhiteBoxXAI

logger = logging.getLogger(__name__)


class CrewAIMonitor:
    """
    Monitor CrewAI multi-agent workflows.

    Automatically tracks:
    - Workflow lifecycle (start, completion)
    - Agent definitions and executions
    - Task assignments and completions
    - Agent-to-agent interactions
    - Token usage and costs

    Example:
        >>> from whiteboxxai.integrations import CrewAIMonitor
        >>> from crewai import Agent, Task, Crew
        >>>
        >>> monitor = CrewAIMonitor(api_key="your_api_key")
        >>>
        >>> # Define agents
        >>> researcher = Agent(
        ...     role="Research Analyst",
        ...     goal="Find accurate information",
        ...     backstory="Expert researcher",
        ...     tools=[search_tool]
        ... )
        >>>
        >>> writer = Agent(
        ...     role="Content Writer",
        ...     goal="Write engaging content",
        ...     backstory="Professional writer",
        ...     tools=[writing_tool]
        ... )
        >>>
        >>> # Create tasks
        >>> research_task = Task(
        ...     description="Research topic X",
        ...     agent=researcher
        ... )
        >>>
        >>> writing_task = Task(
        ...     description="Write article based on research",
        ...     agent=writer
        ... )
        >>>
        >>> # Create and monitor crew
        >>> crew = Crew(
        ...     agents=[researcher, writer],
        ...     tasks=[research_task, writing_task],
        ...     process=Process.sequential
        ... )
        >>>
        >>> workflow_id = monitor.start_monitoring(
        ...     crew=crew,
        ...     workflow_name="Article Generation",
        ...     metadata={"topic": "AI Safety"}
        ... )
        >>>
        >>> # Execute crew (automatically monitored)
        >>> result = crew.kickoff()
        >>>
        >>> # Complete monitoring
        >>> monitor.complete_monitoring(outputs={"article": result})
    """

    def __init__(
        self,
        api_key: str,
        api_url: Optional[str] = None,
        organization_id: Optional[str] = None,
    ):
        """
        Initialize CrewAI monitor.

        Args:
            api_key: WhiteBoxXAI API key
            api_url: WhiteBoxXAI API URL (optional, defaults to production)
            organization_id: Organization ID (optional, extracted from token if not provided)
        """
        self.client = (
            WhiteBoxXAI(api_key=api_key, base_url=api_url)
            if api_url
            else WhiteBoxXAI(api_key=api_key)
        )
        self.organization_id = organization_id
        self.workflow_id = None
        self.agent_map = {}  # Maps CrewAI agent to WhiteBoxXAI agent_id
        self.task_map = {}  # Maps CrewAI task to WhiteBoxXAI task_id
        self.execution_map = {}  # Maps agent to execution_id

        logger.info("CrewAI Monitor initialized")

    def start_monitoring(
        self,
        crew: Any,  # crewai.Crew
        workflow_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start monitoring a CrewAI crew workflow.

        Args:
            crew: CrewAI Crew instance
            workflow_name: Name for the workflow
            metadata: Optional workflow metadata

        Returns:
            Workflow ID (UUID string)
        """
        try:
            # Create workflow
            workflow_data = {
                "name": workflow_name,
                "framework": "crewai",
                "metadata": metadata or {},
            }

            response = self.client.request(
                "POST", "/api/v1/workflows/multi-agent/start", data=workflow_data
            )

            self.workflow_id = response.get("id")
            logger.info(f"Started monitoring CrewAI workflow: {self.workflow_id}")

            # Register agents
            for agent in crew.agents:
                self._register_agent(agent)

            # Register tasks
            for task in crew.tasks:
                self._register_task(task)

            # Start workflow
            start_data = {
                "inputs": {
                    "agent_count": len(crew.agents),
                    "task_count": len(crew.tasks),
                    "process": str(crew.process)
                    if hasattr(crew, "process")
                    else "sequential",
                }
            }

            self.client.request(
                "POST",
                f"/api/v1/workflows/multi-agent/{self.workflow_id}/start",
                data=start_data,
            )

            return self.workflow_id

        except Exception as e:
            logger.error(f"Error starting CrewAI monitoring: {str(e)}")
            raise

    def _register_agent(self, crew_agent: Any) -> str:
        """
        Register a CrewAI agent with WhiteBoxXAI.

        Args:
            crew_agent: CrewAI Agent instance

        Returns:
            Agent ID (UUID string)
        """
        try:
            agent_data = {
                "name": getattr(crew_agent, "role", "Unknown Agent"),
                "role": getattr(crew_agent, "role", None),
                "agent_type": "crewai_agent",
                "goal": getattr(crew_agent, "goal", None),
                "backstory": getattr(crew_agent, "backstory", None),
                "tools": [
                    tool.__class__.__name__ for tool in getattr(crew_agent, "tools", [])
                ],
                "llm_provider": getattr(
                    getattr(crew_agent, "llm", None), "model_name", "unknown"
                ).split("-")[0]
                if hasattr(crew_agent, "llm")
                else None,
                "model_name": getattr(
                    getattr(crew_agent, "llm", None), "model_name", None
                )
                if hasattr(crew_agent, "llm")
                else None,
                "metadata": {
                    "verbose": getattr(crew_agent, "verbose", False),
                    "allow_delegation": getattr(crew_agent, "allow_delegation", False),
                    "max_iter": getattr(crew_agent, "max_iter", None),
                },
            }

            response = self.client.request(
                "POST",
                f"/api/v1/workflows/multi-agent/{self.workflow_id}/agents",
                data=agent_data,
            )

            agent_id = response.get("id")
            self.agent_map[id(crew_agent)] = agent_id

            logger.debug(f"Registered agent: {agent_data['name']} ({agent_id})")

            return agent_id

        except Exception as e:
            logger.error(f"Error registering agent: {str(e)}")
            raise

    def _register_task(self, crew_task: Any) -> str:
        """
        Register a CrewAI task with WhiteBoxXAI.

        Args:
            crew_task: CrewAI Task instance

        Returns:
            Task ID (UUID string)
        """
        try:
            # Get agent ID for task's assigned agent
            agent_id = None
            if hasattr(crew_task, "agent") and crew_task.agent:
                agent_id = self.agent_map.get(id(crew_task.agent))

            task_data = {
                "task_name": getattr(crew_task, "description", "Unknown Task")[:255],
                "description": getattr(crew_task, "description", None),
                "expected_output": getattr(crew_task, "expected_output", None),
                "agent_id": agent_id,
                "context": {
                    "tools": [
                        tool.__class__.__name__
                        for tool in getattr(crew_task, "tools", [])
                    ],
                    "async_execution": getattr(crew_task, "async_execution", False),
                },
            }

            response = self.client.request(
                "POST",
                f"/api/v1/workflows/multi-agent/{self.workflow_id}/tasks",
                data=task_data,
            )

            task_id = response.get("id")
            self.task_map[id(crew_task)] = task_id

            logger.debug(
                f"Registered task: {task_data['task_name'][:50]}... ({task_id})"
            )

            return task_id

        except Exception as e:
            logger.error(f"Error registering task: {str(e)}")
            raise

    def log_agent_execution(
        self,
        agent: Any,
        inputs: Optional[Dict] = None,
        outputs: Optional[Dict] = None,
        tokens_used: int = 0,
        cost: float = 0.0,
    ) -> None:
        """
        Log an agent execution.

        Args:
            agent: CrewAI Agent instance
            inputs: Execution inputs
            outputs: Execution outputs
            tokens_used: Tokens consumed
            cost: Execution cost
        """
        try:
            agent_id = self.agent_map.get(id(agent))
            if not agent_id:
                logger.warning(f"Agent not registered, skipping execution log")
                return

            execution_data = {
                "agent_id": agent_id,
                "inputs": inputs,
                "outputs": outputs,
                "tokens_used": tokens_used,
                "cost": cost,
                "status": "completed",
            }

            self.client.request(
                "POST",
                f"/api/v1/workflows/multi-agent/{self.workflow_id}/executions",
                data=execution_data,
            )

            logger.debug(f"Logged execution for agent: {agent_id}")

        except Exception as e:
            logger.error(f"Error logging agent execution: {str(e)}")

    def log_task_completion(
        self,
        task: Any,
        status: str = "completed",
        output_data: Optional[Dict] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Log task completion.

        Args:
            task: CrewAI Task instance
            status: Task status (completed, failed)
            output_data: Task output
            error_message: Error message if failed
        """
        try:
            task_id = self.task_map.get(id(task))
            if not task_id:
                logger.warning(f"Task not registered, skipping completion log")
                return

            update_data = {
                "status": status,
                "output_data": output_data,
                "error_message": error_message,
            }

            self.client.request(
                "PATCH",
                f"/api/v1/workflows/multi-agent/tasks/{task_id}",
                data=update_data,
            )

            logger.debug(f"Logged task completion: {task_id} ({status})")

        except Exception as e:
            logger.error(f"Error logging task completion: {str(e)}")

    def log_interaction(
        self,
        from_agent: Any,
        to_agent: Any,
        interaction_type: str = "delegation",
        message: Optional[str] = None,
    ) -> None:
        """
        Log agent-to-agent interaction.

        Args:
            from_agent: Source agent
            to_agent: Target agent
            interaction_type: Type of interaction (delegation, handoff, query, feedback)
            message: Interaction message
        """
        try:
            from_agent_id = self.agent_map.get(id(from_agent))
            to_agent_id = self.agent_map.get(id(to_agent))

            if not from_agent_id or not to_agent_id:
                logger.warning("Agents not registered, skipping interaction log")
                return

            interaction_data = {
                "interaction_type": interaction_type,
                "from_agent_id": from_agent_id,
                "to_agent_id": to_agent_id,
                "message": message,
            }

            self.client.request(
                "POST",
                f"/api/v1/workflows/multi-agent/{self.workflow_id}/interactions",
                data=interaction_data,
            )

            logger.debug(
                f"Logged interaction: {from_agent_id} -> {to_agent_id} ({interaction_type})"
            )

        except Exception as e:
            logger.error(f"Error logging interaction: {str(e)}")

    def complete_monitoring(
        self,
        status: str = "completed",
        outputs: Optional[Dict] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Complete workflow monitoring.

        Args:
            status: Final workflow status (completed, failed, cancelled)
            outputs: Workflow outputs
            error_message: Error message if failed

        Returns:
            Workflow summary with analytics
        """
        if not self.workflow_id:
            raise ValueError("No active workflow to complete. Call start_monitoring() first.")

        try:
            complete_data = {
                "status": status,
                "outputs": outputs,
                "error_message": error_message,
            }

            response = self.client.request(
                "POST",
                f"/api/v1/workflows/multi-agent/{self.workflow_id}/complete",
                data=complete_data,
            )

            logger.info(f"Completed monitoring workflow: {self.workflow_id} ({status})")

            # Get analytics
            analytics = self.get_analytics()

            return {
                "workflow_id": self.workflow_id,
                "status": status,
                "analytics": analytics,
            }

        except Exception as e:
            logger.error(f"Error completing monitoring: {str(e)}")
            raise

    def get_analytics(self) -> Dict[str, Any]:
        """
        Get workflow analytics.

        Returns:
            Dict with metrics, cost breakdown, and bottlenecks
        """
        try:
            if not self.workflow_id:
                raise ValueError("No active workflow")

            analytics = self.client.request(
                "GET", f"/api/v1/workflows/multi-agent/{self.workflow_id}/analytics"
            )

            cost_breakdown = self.client.request(
                "GET",
                f"/api/v1/workflows/multi-agent/{self.workflow_id}/cost-breakdown",
            )

            return {"metrics": analytics, "cost_breakdown": cost_breakdown}

        except Exception as e:
            logger.error(f"Error getting analytics: {str(e)}")
            return {}


# Helper function for easy usage
def monitor_crew(
    crew: Any,
    workflow_name: str,
    api_key: str,
    api_url: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> CrewAIMonitor:
    """
    Convenience function to start monitoring a CrewAI crew.

    Args:
        crew: CrewAI Crew instance
        workflow_name: Workflow name
        api_key: WhiteBoxXAI API key
        api_url: WhiteBoxXAI API URL (optional)
        metadata: Workflow metadata (optional)

    Returns:
        CrewAIMonitor instance with workflow started

    Example:
        >>> monitor = monitor_crew(
        ...     crew=my_crew,
        ...     workflow_name="Research & Writing",
        ...     api_key="your_api_key"
        ... )
        >>> result = my_crew.kickoff()
        >>> monitor.complete_monitoring(outputs={"result": result})
    """
    monitor = CrewAIMonitor(api_key=api_key, api_url=api_url)
    monitor.start_monitoring(crew=crew, workflow_name=workflow_name, metadata=metadata)
    return monitor
