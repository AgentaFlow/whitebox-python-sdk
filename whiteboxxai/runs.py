"""Group the records one AI run produces into shared Governance Events.

WhiteBoxXAI bills in Governance Events. An LLM interaction, logged
prediction, RAG retrieval or safety scan each costs one on its own, but
records sent inside ``governance_run()`` carry the same ``X-WhiteBox-Run-Id``
header and share events: up to 100 records per event. A 12-call agent run is
one event, however many LLM calls and tool steps it logs.

Example:
    ```python
    from whiteboxxai import governance_run

    with governance_run():
        client.llm.log_call(...)
        client.predictions.log(...)
    ```

The LangChain callback handler does this automatically, keyed on the chain's
root run. Agent workflows (``client.agent_workflows``, CrewAI, LangChain
agents) need nothing: a workflow is one event by itself.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

RUN_ID_HEADER = "X-WhiteBox-Run-Id"

_current_run_id: ContextVar[str | None] = ContextVar("whiteboxxai_governance_run_id", default=None)


def current_run_id() -> str | None:
    """The run id requests are being grouped under right now, if any."""
    return _current_run_id.get()


@contextmanager
def governance_run(run_id: str | None = None) -> Iterator[str]:
    """Send every request made inside the block as part of one run.

    Args:
        run_id: An opaque id of up to 128 letters, digits, ``.``, ``_``,
            ``:`` or ``-``. Pass your own to group records across processes;
            a random one is generated otherwise. The server ignores an id
            outside that shape, and those records are billed individually.

    Yields:
        The run id in effect.
    """
    run_id = run_id or uuid.uuid4().hex
    token = _current_run_id.set(run_id)
    try:
        yield run_id
    finally:
        _current_run_id.reset(token)


def with_run_header(kwargs: dict) -> dict:
    """Add the run header to an httpx request's kwargs when a run is active.

    An explicit ``headers`` value from the caller wins over the ambient run.
    """
    run_id = current_run_id()
    if run_id is None:
        return kwargs
    headers = dict(kwargs.get("headers") or {})
    headers.setdefault(RUN_ID_HEADER, run_id)
    return {**kwargs, "headers": headers}
