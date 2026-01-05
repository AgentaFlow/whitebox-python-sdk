"""
LangChain Integration

Integration for monitoring LangChain applications including chains, agents, and RAG pipelines.
"""

from typing import Any, Dict, Optional, List
import warnings
import time

try:
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import LLMResult, AgentAction, AgentFinish
    from langchain.chains.base import Chain
    from langchain.agents import AgentExecutor
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object
    Chain = object
    AgentExecutor = object

from explainai.monitor import ModelMonitor


class LangChainMonitor(ModelMonitor):
    """
    Monitor for LangChain applications.

    Supports:
    - Chain executions
    - Agent runs
    - LLM calls
    - Tool usage
    - RAG pipelines
    - Memory operations

    Example:
        ```python
        from langchain.chains import LLMChain
        from langchain.llms import OpenAI
        from whiteboxai import WhiteBoxAI
        from explainai.integrations.langchain import LangChainMonitor

        # Setup monitoring
        client = WhiteBoxAI(api_key="your-api-key")
        monitor = LangChainMonitor(
            client=client,
            application_name="my_langchain_app"
        )

        # Register application
        monitor.register_application(
            name="LangChain Q&A",
            version="1.0.0"
        )

        # Create callback handler
        callback = monitor.create_callback_handler()

        # Use with chain
        llm = OpenAI()
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(input="Question", callbacks=[callback])
        ```
    """

    def __init__(
        self,
        client,
        application_name: Optional[str] = None,
        track_tokens: bool = True,
        track_cost: bool = True,
        **kwargs
    ):
        """
        Initialize LangChain monitor.

        Args:
            client: WhiteBoxAI client instance
            application_name: Name of the LangChain application
            track_tokens: Track token usage
            track_cost: Track API costs
            **kwargs: Additional arguments for ModelMonitor
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain is not installed. Install with: pip install langchain"
            )

        super().__init__(client, **kwargs)
        self._application_name = application_name
        self._track_tokens = track_tokens
        self._track_cost = track_cost
        self._registered = False

    def register_application(
        self,
        name: Optional[str] = None,
        version: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        """
        Register LangChain application with WhiteBoxAI.

        Args:
            name: Application name
            version: Application version
            description: Application description
            **kwargs: Additional metadata

        Returns:
            Model ID (application ID)
        """
        if name is None:
            name = self._application_name or "langchain_app"

        # Prepare metadata
        metadata = {
            "framework": "langchain",
            "application_type": "langchain",
            "track_tokens": self._track_tokens,
            "track_cost": self._track_cost,
            **kwargs,
        }

        # Register as a model (application)
        self.model_id = self.client.register_model(
            name=name,
            model_type="generation",  # LangChain apps are typically generative
            framework="langchain",
            version=version,
            description=description,
            metadata=metadata,
        )

        self._registered = True
        return self.model_id

    def create_callback_handler(self) -> "WhiteBoxAICallbackHandler":
        """
        Create a LangChain callback handler for automatic logging.

        Returns:
            WhiteBoxAICallbackHandler instance

        Example:
            ```python
            callback = monitor.create_callback_handler()
            chain.run(input="Question", callbacks=[callback])
            ```
        """
        if not self._registered:
            self.register_application()

        return WhiteBoxAICallbackHandler(monitor=self)

    def log_chain_execution(
        self,
        chain_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        execution_time: float,
        llm_calls: Optional[List[Dict]] = None,
        tool_calls: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log a chain execution.

        Args:
            chain_name: Name of the chain
            inputs: Chain inputs
            outputs: Chain outputs
            execution_time: Time taken to execute (seconds)
            llm_calls: List of LLM calls made
            tool_calls: List of tool calls made
            **kwargs: Additional metadata
        """
        metadata = {
            "chain_name": chain_name,
            "execution_time": execution_time,
            "num_llm_calls": len(llm_calls) if llm_calls else 0,
            "num_tool_calls": len(tool_calls) if tool_calls else 0,
            **kwargs,
        }

        if llm_calls:
            metadata["llm_calls"] = llm_calls
        if tool_calls:
            metadata["tool_calls"] = tool_calls

        # Log as prediction
        self.log_prediction(
            inputs=inputs,
            prediction=outputs,
            metadata=metadata,
        )

    def log_agent_execution(
        self,
        agent_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        execution_time: float,
        steps: List[Dict],
        **kwargs: Any,
    ) -> None:
        """
        Log an agent execution.

        Args:
            agent_name: Name of the agent
            inputs: Agent inputs
            outputs: Agent outputs
            execution_time: Time taken to execute (seconds)
            steps: List of agent steps/actions
            **kwargs: Additional metadata
        """
        metadata = {
            "agent_name": agent_name,
            "execution_time": execution_time,
            "num_steps": len(steps),
            "steps": steps,
            **kwargs,
        }

        # Log as prediction
        self.log_prediction(
            inputs=inputs,
            prediction=outputs,
            metadata=metadata,
        )

    def log_llm_call(
        self,
        prompt: str,
        response: str,
        model: str,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        latency: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log an LLM call.

        Args:
            prompt: Input prompt
            response: LLM response
            model: Model name
            tokens_used: Number of tokens used
            cost: API cost
            latency: Response latency (seconds)
            **kwargs: Additional metadata
        """
        metadata = {
            "model": model,
            "tokens_used": tokens_used,
            "cost": cost,
            "latency": latency,
            **kwargs,
        }

        self.log_prediction(
            inputs={"prompt": prompt},
            prediction={"response": response},
            metadata=metadata,
        )

    def log_tool_call(
        self,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
        execution_time: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log a tool call.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input
            tool_output: Tool output
            execution_time: Time taken to execute (seconds)
            **kwargs: Additional metadata
        """
        metadata = {
            "tool_name": tool_name,
            "execution_time": execution_time,
            **kwargs,
        }

        self.log_prediction(
            inputs={"tool_input": tool_input},
            prediction={"tool_output": tool_output},
            metadata=metadata,
        )

    def log_rag_retrieval(
        self,
        query: str,
        documents: List[Dict],
        num_retrieved: int,
        retrieval_time: Optional[float] = None,
        relevance_scores: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log a RAG retrieval operation.

        Args:
            query: Query text
            documents: Retrieved documents
            num_retrieved: Number of documents retrieved
            retrieval_time: Time taken to retrieve (seconds)
            relevance_scores: Relevance scores for documents
            **kwargs: Additional metadata
        """
        metadata = {
            "num_retrieved": num_retrieved,
            "retrieval_time": retrieval_time,
            "relevance_scores": relevance_scores,
            **kwargs,
        }

        self.log_prediction(
            inputs={"query": query},
            prediction={"documents": documents},
            metadata=metadata,
        )


class WhiteBoxAICallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler for automatic logging to WhiteBoxAI.

    Example:
        ```python
        monitor = LangChainMonitor(client, application_name="my_app")
        monitor.register_application()
        callback = monitor.create_callback_handler()

        # Use with chain
        chain.run(input="Question", callbacks=[callback])

        # Use with agent
        agent.run(input="Task", callbacks=[callback])
        ```
    """

    def __init__(self, monitor: LangChainMonitor):
        """
        Initialize callback handler.

        Args:
            monitor: LangChainMonitor instance
        """
        super().__init__()
        self.monitor = monitor
        self._chain_start_times = {}
        self._agent_start_times = {}
        self._llm_start_times = {}
        self._tool_start_times = {}
        self._chain_inputs = {}
        self._agent_inputs = {}
        self._agent_steps = {}

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts running."""
        run_id = kwargs.get("run_id", id(serialized))
        self._chain_start_times[run_id] = time.time()
        self._chain_inputs[run_id] = inputs

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Called when a chain ends running."""
        run_id = kwargs.get("run_id")
        if run_id in self._chain_start_times:
            execution_time = time.time() - self._chain_start_times[run_id]
            inputs = self._chain_inputs.get(run_id, {})

            # Get chain name
            chain_name = kwargs.get("name", "unknown_chain")

            # Log chain execution
            try:
                self.monitor.log_chain_execution(
                    chain_name=chain_name,
                    inputs=inputs,
                    outputs=outputs,
                    execution_time=execution_time,
                )
            except Exception as e:
                warnings.warn(f"Failed to log chain execution: {e}")

            # Cleanup
            del self._chain_start_times[run_id]
            if run_id in self._chain_inputs:
                del self._chain_inputs[run_id]

    def on_chain_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """Called when a chain errors."""
        run_id = kwargs.get("run_id")
        if run_id in self._chain_start_times:
            # Cleanup on error
            del self._chain_start_times[run_id]
            if run_id in self._chain_inputs:
                del self._chain_inputs[run_id]

    def on_agent_action(
        self,
        action: AgentAction,
        **kwargs: Any,
    ) -> None:
        """Called when an agent takes an action."""
        run_id = kwargs.get("run_id")
        if run_id not in self._agent_steps:
            self._agent_steps[run_id] = []

        self._agent_steps[run_id].append({
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log,
        })

    def on_agent_finish(
        self,
        finish: AgentFinish,
        **kwargs: Any,
    ) -> None:
        """Called when an agent finishes running."""
        run_id = kwargs.get("run_id")
        if run_id in self._agent_start_times:
            execution_time = time.time() - self._agent_start_times[run_id]
            inputs = self._agent_inputs.get(run_id, {})
            steps = self._agent_steps.get(run_id, [])

            # Log agent execution
            try:
                self.monitor.log_agent_execution(
                    agent_name=kwargs.get("name", "unknown_agent"),
                    inputs=inputs,
                    outputs=finish.return_values,
                    execution_time=execution_time,
                    steps=steps,
                )
            except Exception as e:
                warnings.warn(f"Failed to log agent execution: {e}")

            # Cleanup
            del self._agent_start_times[run_id]
            if run_id in self._agent_inputs:
                del self._agent_inputs[run_id]
            if run_id in self._agent_steps:
                del self._agent_steps[run_id]

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Called when an LLM starts generating."""
        run_id = kwargs.get("run_id", id(serialized))
        self._llm_start_times[run_id] = time.time()

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM ends generating."""
        run_id = kwargs.get("run_id")
        if run_id in self._llm_start_times:
            latency = time.time() - self._llm_start_times[run_id]

            # Extract response details
            if response.generations and len(response.generations) > 0:
                generation = response.generations[0][0]
                response_text = generation.text

                # Extract token usage if available
                tokens_used = None
                if response.llm_output and "token_usage" in response.llm_output:
                    tokens_used = response.llm_output["token_usage"].get("total_tokens")

                # Log LLM call (if configured)
                if self.monitor._track_tokens or self.monitor._track_cost:
                    try:
                        self.monitor.log_llm_call(
                            prompt="",  # Prompt tracked separately
                            response=response_text,
                            model=kwargs.get("invocation_params", {}).get("model_name", "unknown"),
                            tokens_used=tokens_used,
                            latency=latency,
                        )
                    except Exception as e:
                        warnings.warn(f"Failed to log LLM call: {e}")

            # Cleanup
            del self._llm_start_times[run_id]

    def on_llm_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM errors."""
        run_id = kwargs.get("run_id")
        if run_id in self._llm_start_times:
            del self._llm_start_times[run_id]

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts running."""
        run_id = kwargs.get("run_id", id(serialized))
        self._tool_start_times[run_id] = time.time()

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> None:
        """Called when a tool ends running."""
        run_id = kwargs.get("run_id")
        if run_id in self._tool_start_times:
            execution_time = time.time() - self._tool_start_times[run_id]

            # Log tool call
            try:
                self.monitor.log_tool_call(
                    tool_name=kwargs.get("name", "unknown_tool"),
                    tool_input=kwargs.get("input_str", ""),
                    tool_output=output,
                    execution_time=execution_time,
                )
            except Exception as e:
                warnings.warn(f"Failed to log tool call: {e}")

            # Cleanup
            del self._tool_start_times[run_id]

    def on_tool_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """Called when a tool errors."""
        run_id = kwargs.get("run_id")
        if run_id in self._tool_start_times:
            del self._tool_start_times[run_id]


def wrap_langchain_chain(
    chain: Chain,
    monitor: LangChainMonitor,
) -> Chain:
    """
    Wrap a LangChain chain to automatically log executions.

    Args:
        chain: LangChain chain to wrap
        monitor: LangChainMonitor instance

    Returns:
        Wrapped chain that logs executions

    Example:
        ```python
        chain = LLMChain(llm=llm, prompt=prompt)
        wrapped_chain = wrap_langchain_chain(chain, monitor)

        # Executions automatically logged
        result = wrapped_chain.run(input="Question")
        ```
    """
    if not monitor._registered:
        monitor.register_application()

    # Create callback handler
    callback = monitor.create_callback_handler()

    # Store original run method
    original_run = chain.run
    original_call = chain.__call__

    def logged_run(*args, **kwargs):
        # Add callback to kwargs
        if "callbacks" not in kwargs:
            kwargs["callbacks"] = []
        kwargs["callbacks"].append(callback)

        # Run chain
        return original_run(*args, **kwargs)

    def logged_call(*args, **kwargs):
        # Add callback to kwargs
        if "callbacks" not in kwargs:
            kwargs["callbacks"] = []
        kwargs["callbacks"].append(callback)

        # Call chain
        return original_call(*args, **kwargs)

    # Replace methods
    chain.run = logged_run
    chain.__call__ = logged_call

    return chain


__all__ = [
    'LangChainMonitor',
    'WhiteBoxAICallbackHandler',
    'wrap_langchain_chain',
]
