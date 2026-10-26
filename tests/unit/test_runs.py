"""Tests for governance_run: grouping one AI run's records into shared events (#822)."""

import asyncio
import uuid
from types import SimpleNamespace

import httpx
import pytest

from whiteboxxai import WhiteBoxXAI, governance_run
from whiteboxxai.integrations.langchain import WhiteBoxXAICallbackHandler
from whiteboxxai.runs import RUN_ID_HEADER, current_run_id, with_run_header


class TestGovernanceRun:
    def test_sets_and_restores_the_run_id(self):
        assert current_run_id() is None
        with governance_run("outer") as outer:
            assert outer == "outer"
            with governance_run("inner"):
                assert current_run_id() == "inner"
            assert current_run_id() == "outer"
        assert current_run_id() is None

    def test_generates_an_id_when_none_given(self):
        with governance_run() as run_id:
            assert len(run_id) == 32
            assert current_run_id() == run_id

    def test_with_run_header_is_a_noop_outside_a_run(self):
        kwargs = {"timeout": 5}
        assert with_run_header(kwargs) is kwargs

    def test_with_run_header_adds_the_header(self):
        with governance_run("r1"):
            assert with_run_header({})["headers"] == {RUN_ID_HEADER: "r1"}

    def test_explicit_header_wins_and_is_not_mutated(self):
        mine = {RUN_ID_HEADER: "mine", "X-Other": "1"}
        with governance_run("ambient"):
            sent = with_run_header({"headers": mine})["headers"]
        assert sent[RUN_ID_HEADER] == "mine"
        assert mine == {RUN_ID_HEADER: "mine", "X-Other": "1"}


def _recording_client(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get(RUN_ID_HEADER))
        return httpx.Response(200, json={"ok": True})

    client = WhiteBoxXAI(api_key="test_key")
    transport = httpx.MockTransport(handler)
    client._sync_client = httpx.Client(base_url=client.config.base_url, transport=transport)
    client._async_client = httpx.AsyncClient(base_url=client.config.base_url, transport=transport)
    return client


class TestClientSendsTheHeader:
    def test_sync_requests_inside_a_run_carry_it(self):
        seen: list = []
        client = _recording_client(seen)
        client.request("POST", "/api/v1/llm/logs", data={})
        with governance_run("run-7"):
            client.request("POST", "/api/v1/llm/logs", data={})
        assert seen == [None, "run-7"]

    def test_async_requests_inside_a_run_carry_it(self):
        seen: list = []
        client = _recording_client(seen)

        async def go():
            with governance_run("run-8"):
                await client.arequest("POST", "/api/v1/llm/logs", data={})

        asyncio.run(go())
        assert seen == ["run-8"]


class _Monitor:
    """Stands in for LangChainMonitor and records the run each log was sent under."""

    _track_tokens = True
    _track_cost = False

    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []

    def _record(self, kind):
        def log(**_kwargs):
            self.calls.append((kind, current_run_id()))

        return log

    def __getattr__(self, name):
        if name.startswith("log_"):
            return self._record(name)
        raise AttributeError(name)


def _llm_result(text="answer"):
    generation = SimpleNamespace(text=text, message=None)
    return SimpleNamespace(generations=[[generation]], llm_output=None)


class TestLangChainHandlerGroupsARun:
    def test_everything_one_invocation_logs_shares_its_root_run(self):
        monitor = _Monitor()
        handler = WhiteBoxXAICallbackHandler(monitor)
        root, llm, tool = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

        handler.on_chain_start({}, {"q": "x"}, run_id=root, parent_run_id=None)
        handler.on_llm_start({}, ["p"], run_id=llm, parent_run_id=root)
        handler.on_llm_end(_llm_result(), run_id=llm, parent_run_id=root)
        handler.on_tool_start({}, "in", run_id=tool, parent_run_id=root)
        handler.on_tool_end("out", run_id=tool, parent_run_id=root, name="search")
        handler.on_chain_end({"a": "y"}, run_id=root, parent_run_id=None)

        assert [kind for kind, _ in monitor.calls] == [
            "log_llm_call",
            "log_tool_call",
            "log_chain_execution",
        ]
        assert {run for _, run in monitor.calls} == {str(root)}

    def test_nested_chains_resolve_to_the_outermost_root(self):
        monitor = _Monitor()
        handler = WhiteBoxXAICallbackHandler(monitor)
        root, child, llm = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

        handler.on_chain_start({}, {}, run_id=root, parent_run_id=None)
        handler.on_chain_start({}, {}, run_id=child, parent_run_id=root)
        handler.on_llm_start({}, ["p"], run_id=llm, parent_run_id=child)
        handler.on_llm_end(_llm_result(), run_id=llm, parent_run_id=child)

        assert monitor.calls == [("log_llm_call", str(root))]

    def test_separate_invocations_are_separate_runs_and_state_is_released(self):
        monitor = _Monitor()
        handler = WhiteBoxXAICallbackHandler(monitor)
        first, second = uuid.uuid4(), uuid.uuid4()

        for root in (first, second):
            handler.on_chain_start({}, {}, run_id=root, parent_run_id=None)
            handler.on_chain_end({}, run_id=root, parent_run_id=None)

        assert [run for _, run in monitor.calls] == [str(first), str(second)]
        assert handler._run_roots == {}

    @pytest.mark.parametrize("event", ["on_chain_error", "on_llm_error", "on_tool_error"])
    def test_errors_release_the_run(self, event):
        handler = WhiteBoxXAICallbackHandler(_Monitor())
        run = uuid.uuid4()
        handler.on_chain_start({}, {}, run_id=run, parent_run_id=None)
        getattr(handler, event)(RuntimeError("x"), run_id=run)
        assert run not in handler._run_roots
