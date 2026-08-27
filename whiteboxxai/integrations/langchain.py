"""
LangChain Integration

Integration for monitoring LangChain applications including chains, agents, and RAG pipelines.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional

# langchain_core is the modern, lighter-weight home for these types and may
# be installed without the full legacy langchain package. Try it first, then
# fall back to the legacy langchain.callbacks.base/langchain.schema locations.
try:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
        from langchain.schema import AgentAction, AgentFinish, LLMResult

        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        BaseCallbackHandler = object
        AgentAction = object
        AgentFinish = object
        LLMResult = object

# AgentExecutor/Chain are higher-level `langchain` package abstractions, not
# part of langchain_core, so they may be unavailable even when langchain_core
# (and thus LANGCHAIN_AVAILABLE) is -- probe them separately.
try:
    from langchain.agents import AgentExecutor
except ImportError:
    AgentExecutor = object

try:
    from langchain.chains.base import Chain
except ImportError:
    Chain = object

from whiteboxxai.monitor import ModelMonitor


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
        from whiteboxxai import WhiteBoxXAI
        from whiteboxxai.integrations.langchain import LangChainMonitor

        # Setup monitoring
        client = WhiteBoxXAI(api_key="your-api-key")
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
        **kwargs,
    ):
        """
        Initialize LangChain monitor.

        Args:
            client: WhiteBoxXAI client instance
            application_name: Name of the LangChain application
            track_tokens: Track token usage
            track_cost: Track API costs
            **kwargs: Additional arguments for ModelMonitor
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain is not installed. Install with: pip install langchain")

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
        Register LangChain application with WhiteBoxXAI.

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

    def create_callback_handler(self) -> "WhiteBoxXAICallbackHandler":
        """
        Create a LangChain callback handler for automatic logging.

        Returns:
            WhiteBoxXAICallbackHandler instance

        Example:
            ```python
            callback = monitor.create_callback_handler()
            chain.run(input="Question", callbacks=[callback])
            ```
        """
        if not self._registered:
            self.register_application()

        return WhiteBoxXAICallbackHandler(monitor=self)

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
            output=outputs,
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
            output=outputs,
            metadata=metadata,
        )

    def log_llm_call(
        self,
        prompt: str,
        response: str,
        model: str,
        provider: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        tokens_used: Optional[int] = None,
        latency: Optional[float] = None,
        finish_reason: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log an LLM call to WhiteBoxXAI's real LLM monitoring endpoint
        (POST /api/v1/llm/logs), instead of the generic prediction-logging
        endpoint -- token/cost/latency data logged this way is queryable via
        client.llm.get_stats() and shows up on the LLM observability
        dashboard, rather than sitting as opaque metadata on a prediction
        record. Cost is computed server-side from tokens (see
        backend/llm/token_counter.py's pricing table), so it's not accepted
        here.

        Args:
            prompt: Input prompt
            response: LLM response
            model: Model name
            provider: LLM provider name (openai, anthropic, etc.); defaults
                to "langchain" when the specific provider isn't known
            prompt_tokens: Number of tokens in the prompt, if known
            completion_tokens: Number of tokens in the completion, if known
            tokens_used: Legacy combined token count -- used as
                completion_tokens when completion_tokens isn't given
                separately
            latency: Response latency, in seconds
            finish_reason: Why the LLM stopped (stop, length, etc.)
            **kwargs: Additional data, stored as response metadata
        """
        self.client.llm.log_call(
            provider=provider or "langchain",
            model_name=model,
            prompt=prompt,
            completion=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=(completion_tokens if completion_tokens is not None else tokens_used),
            latency_ms=(latency * 1000) if latency is not None else 0.0,
            finish_reason=finish_reason,
            model_id=self.model_id,
            response_metadata=kwargs or None,
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
            output={"tool_output": tool_output},
            metadata=metadata,
        )

    def log_rag_retrieval(
        self,
        query: str,
        documents: List[Dict],
        num_retrieved: int,
        retrieval_time: Optional[float] = None,
        relevance_scores: Optional[List[float]] = None,
        ground_truth_ids: Optional[List[str]] = None,
        answer: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log a RAG retrieval operation to WhiteBoxXAI's real RAG monitoring
        endpoint (POST /api/v1/rag/retrievals), instead of the generic
        prediction-logging endpoint. When ground_truth_ids is supplied, the
        backend computes real precision/recall/MRR/NDCG at log time.

        Args:
            query: Query text
            documents: Retrieved documents; each dict may carry "id"
                (falls back to its index if absent), "text"/"content", and
                "score"/"metadata"
            num_retrieved: Number of documents retrieved (used as top_k)
            retrieval_time: Time taken to retrieve, in seconds
            relevance_scores: Per-document relevance scores, same order as
                documents -- overrides each document's own "score" when given
            ground_truth_ids: Relevant document IDs, to compute precision/
                recall/MRR/NDCG at log time
            answer: Generated answer, if a generation step was included
            **kwargs: Additional data, stored as retrieval metadata
        """
        results = []
        for idx, doc in enumerate(documents):
            score = (
                relevance_scores[idx]
                if relevance_scores and idx < len(relevance_scores)
                else doc.get("score", 0.0)
            )
            results.append(
                {
                    "document_id": str(doc.get("id", idx)),
                    "document_content": doc.get("text") or doc.get("content"),
                    "document_metadata": doc.get("metadata"),
                    "rank": idx + 1,
                    "score": score,
                }
            )

        self.client.rag.log_retrieval(
            query=query,
            results=results,
            top_k=num_retrieved,
            retrieval_method="langchain",
            retrieval_latency_ms=(retrieval_time * 1000 if retrieval_time is not None else None),
            ground_truth_ids=ground_truth_ids,
            answer=answer,
            model_id=self.model_id,
            metadata=kwargs or None,
        )


class WhiteBoxXAICallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler for automatic logging to WhiteBoxXAI.

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
                warnings.warn(f"Failed to log chain execution: {e}", stacklevel=2)

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

        self._agent_steps[run_id].append(
            {
                "tool": action.tool,
                "tool_input": action.tool_input,
                "log": action.log,
            }
        )

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
                warnings.warn(f"Failed to log agent execution: {e}", stacklevel=2)

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

                # Extract real prompt/completion token counts, provider-
                # agnostically. Prefer LangChain's modern usage_metadata,
                # which every current chat model (OpenAI, Anthropic, Gemini,
                # ...) populates the same way on generation.message -- unlike
                # the legacy llm_output["token_usage"] shape below, which is
                # OpenAI-specific and absent for other providers. Fall back
                # to that legacy shape only for non-chat (plain Generation,
                # no .message) LLMs. If neither is present, pass None
                # through rather than guessing -- POST /llm/logs already
                # estimates prompt tokens server-side when they're absent.
                prompt_tokens = None
                completion_tokens = None
                usage_metadata = getattr(
                    getattr(generation, "message", None), "usage_metadata", None
                )
                if usage_metadata:
                    prompt_tokens = usage_metadata.get("input_tokens")
                    completion_tokens = usage_metadata.get("output_tokens")
                elif response.llm_output and "token_usage" in response.llm_output:
                    token_usage = response.llm_output["token_usage"]
                    prompt_tokens = token_usage.get("prompt_tokens")
                    completion_tokens = token_usage.get("completion_tokens")

                # Log LLM call (if configured)
                if self.monitor._track_tokens or self.monitor._track_cost:
                    try:
                        self.monitor.log_llm_call(
                            prompt="",  # Prompt tracked separately
                            response=response_text,
                            model=kwargs.get("invocation_params", {}).get("model_name", "unknown"),
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            latency=latency,
                        )
                    except Exception as e:
                        warnings.warn(f"Failed to log LLM call: {e}", stacklevel=2)

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
                warnings.warn(f"Failed to log tool call: {e}", stacklevel=2)

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
    "LangChainMonitor",
    "WhiteBoxXAICallbackHandler",
    "wrap_langchain_chain",
]
