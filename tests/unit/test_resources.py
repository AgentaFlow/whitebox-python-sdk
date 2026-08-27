"""Tests for the SDK resource wrapper classes (sdk/whiteboxxai/resources.py).

Every method here is a thin, uniform delegation to client.request()/
client.arequest() (or, for the four write methods that support offline
queueing, client._request_or_queue()) -- there's no business logic to
break, so these tests exist purely to exercise the delegation itself
(correct HTTP method/endpoint, correct passthrough of the mocked response)
rather than to catch subtle bugs. This module previously had zero direct
test coverage.
"""

from unittest.mock import AsyncMock, patch

import pytest

from whiteboxxai.client import WhiteBoxXAI

SENTINEL = {"ok": True}


@pytest.fixture
def client() -> WhiteBoxXAI:
    return WhiteBoxXAI(api_key="test_key")


@pytest.fixture
def mock_request(client):
    with patch.object(client, "request", return_value=SENTINEL) as m:
        yield m


@pytest.fixture
def mock_arequest(client):
    with patch.object(client, "arequest", new_callable=AsyncMock) as m:
        m.return_value = SENTINEL
        yield m


class TestModelsResource:
    def test_register(self, client, mock_request):
        assert client.models.register("m", "classification") == SENTINEL
        mock_request.assert_called_once()
        assert mock_request.call_args[0][:2] == ("POST", "/api/v1/models/")

    async def test_aregister(self, client, mock_arequest):
        assert await client.models.aregister("m", "classification") == SENTINEL
        mock_arequest.assert_awaited_once()

    def test_register_auto_detect_git_no_repo(self, client, mock_request):
        # No git repo in the test sandbox -> detect_git_context() returns
        # None -> logs a warning and proceeds without git metadata.
        assert client.models.register("m", "classification", auto_detect_git=True)

    def test_get(self, client, mock_request):
        assert client.models.get("m1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/models/m1")

    async def test_aget(self, client, mock_arequest):
        assert await client.models.aget("m1") == SENTINEL

    def test_list(self, client, mock_request):
        assert client.models.list(status_filter="active", tags="a,b") == SENTINEL
        mock_request.assert_called_once()

    async def test_alist(self, client, mock_arequest):
        assert await client.models.alist() == SENTINEL

    def test_update(self, client, mock_request):
        assert client.models.update("m1", name="renamed") == SENTINEL
        mock_request.assert_called_once_with("PATCH", "/api/v1/models/m1", data={"name": "renamed"})

    async def test_aupdate(self, client, mock_arequest):
        assert await client.models.aupdate("m1", name="renamed") == SENTINEL

    def test_update_status(self, client, mock_request):
        assert client.models.update_status("m1", "ARCHIVED") == SENTINEL

    async def test_aupdate_status(self, client, mock_arequest):
        assert await client.models.aupdate_status("m1", "ARCHIVED") == SENTINEL

    def test_get_versions(self, client, mock_request):
        assert client.models.get_versions("fraud-detector") == SENTINEL

    async def test_aget_versions(self, client, mock_arequest):
        assert await client.models.aget_versions("fraud-detector") == SENTINEL

    def test_get_latest(self, client, mock_request):
        assert client.models.get_latest("fraud-detector") == SENTINEL

    async def test_aget_latest(self, client, mock_arequest):
        assert await client.models.aget_latest("fraud-detector") == SENTINEL

    def test_update_baseline(self, client, mock_request):
        result = client.models.update_baseline(
            "m1", {"accuracy": 0.9}, baseline_data_hash="h", baseline_data_count=100
        )
        assert result == SENTINEL

    async def test_aupdate_baseline(self, client, mock_arequest):
        result = await client.models.aupdate_baseline("m1", {"accuracy": 0.9})
        assert result == SENTINEL

    def test_archive(self, client, mock_request):
        assert client.models.archive("m1") == SENTINEL

    async def test_aarchive(self, client, mock_arequest):
        assert await client.models.aarchive("m1") == SENTINEL

    def test_restore(self, client, mock_request):
        assert client.models.restore("m1") == SENTINEL

    async def test_arestore(self, client, mock_arequest):
        assert await client.models.arestore("m1") == SENTINEL

    def test_delete(self, client, mock_request):
        assert client.models.delete("m1") == SENTINEL

    async def test_adelete(self, client, mock_arequest):
        assert await client.models.adelete("m1") == SENTINEL


class TestPredictionsResource:
    def test_log(self, client, mock_request):
        result = client.predictions.log(
            "m1",
            {"x": 1},
            {"y": 0},
            prediction_id="p1",
            latency_ms=12.5,
            metadata={"k": "v"},
        )
        assert result == SENTINEL

    async def test_alog(self, client, mock_arequest):
        result = await client.predictions.alog("m1", {"x": 1}, {"y": 0})
        assert result == SENTINEL

    def test_log_batch(self, client, mock_request):
        result = client.predictions.log_batch("m1", [{"input_data": {}, "output_data": {}}])
        assert result == SENTINEL

    async def test_alog_batch(self, client, mock_arequest):
        result = await client.predictions.alog_batch("m1", [])
        assert result == SENTINEL

    def test_get(self, client, mock_request):
        assert client.predictions.get("p1") == SENTINEL

    async def test_aget(self, client, mock_arequest):
        assert await client.predictions.aget("p1") == SENTINEL

    def test_query(self, client, mock_request):
        result = client.predictions.query(
            "m1", start_time="2026-01-01T00:00:00Z", end_time="2026-01-02T00:00:00Z"
        )
        assert result == SENTINEL

    async def test_aquery(self, client, mock_arequest):
        assert await client.predictions.aquery("m1") == SENTINEL

    def test_get_stats(self, client, mock_request):
        result = client.predictions.get_stats("m1", start_time="2026-01-01T00:00:00Z")
        assert result == SENTINEL

    async def test_aget_stats(self, client, mock_arequest):
        assert await client.predictions.aget_stats("m1") == SENTINEL

    def test_get_recent(self, client, mock_request):
        assert client.predictions.get_recent("m1", limit=5) == SENTINEL

    async def test_aget_recent(self, client, mock_arequest):
        assert await client.predictions.aget_recent("m1") == SENTINEL


class TestExplanationsResource:
    def test_generate(self, client, mock_request):
        assert client.explanations.generate(1, method="lime") == SENTINEL

    async def test_agenerate(self, client, mock_arequest):
        assert await client.explanations.agenerate(1) == SENTINEL

    def test_get(self, client, mock_request):
        assert client.explanations.get(1) == SENTINEL

    async def test_aget(self, client, mock_arequest):
        assert await client.explanations.aget(1) == SENTINEL

    def test_generate_async(self, client, mock_request):
        assert client.explanations.generate_async("m1", {"age": 42}) == SENTINEL
        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/explanations/generate/async",
            data={
                "model_id": "m1",
                "instance": {"age": 42},
                "method": None,
                "prediction_id": None,
                "num_features": 10,
                "num_samples": 5000,
            },
        )

    async def test_agenerate_async(self, client, mock_arequest):
        result = await client.explanations.agenerate_async("m1", {"age": 42})
        assert result == SENTINEL
        mock_arequest.assert_awaited_once_with(
            "POST",
            "/api/v1/explanations/generate/async",
            data={
                "model_id": "m1",
                "instance": {"age": 42},
                "method": None,
                "prediction_id": None,
                "num_features": 10,
                "num_samples": 5000,
            },
        )


class TestDriftResource:
    def test_detect(self, client, mock_request):
        assert client.drift.detect("m1", feature_names=["a", "b"]) == SENTINEL

    async def test_adetect(self, client, mock_arequest):
        assert await client.drift.adetect("m1") == SENTINEL

    def test_create_report(self, client, mock_request):
        assert client.drift.create_report("m1") == SENTINEL

    async def test_acreate_report(self, client, mock_arequest):
        assert await client.drift.acreate_report("m1") == SENTINEL

    def test_get_reports(self, client, mock_request):
        assert client.drift.get_reports("m1") == SENTINEL

    async def test_aget_reports(self, client, mock_arequest):
        assert await client.drift.aget_reports("m1") == SENTINEL

    def test_get_report(self, client, mock_request):
        assert client.drift.get_report("m1", "r1") == SENTINEL

    async def test_aget_report(self, client, mock_arequest):
        assert await client.drift.aget_report("m1", "r1") == SENTINEL

    def test_get_trend(self, client, mock_request):
        assert client.drift.get_trend("m1") == SENTINEL

    async def test_aget_trend(self, client, mock_arequest):
        assert await client.drift.aget_trend("m1") == SENTINEL


class TestFairnessResource:
    def _audit_kwargs(self):
        return dict(
            model_id="m1",
            sensitive_attributes=["gender"],
            y_true=[1, 0],
            y_pred=[1, 1],
            group_data={"gender": ["M", "F"]},
        )

    def test_audit(self, client, mock_request):
        assert client.fairness.audit(**self._audit_kwargs()) == SENTINEL

    async def test_aaudit(self, client, mock_arequest):
        assert await client.fairness.aaudit(**self._audit_kwargs()) == SENTINEL

    def test_get_audit(self, client, mock_request):
        assert client.fairness.get_audit("a1") == SENTINEL

    async def test_aget_audit(self, client, mock_arequest):
        assert await client.fairness.aget_audit("a1") == SENTINEL

    def test_list_audits(self, client, mock_request):
        assert client.fairness.list_audits(model_id="m1") == SENTINEL

    async def test_alist_audits(self, client, mock_arequest):
        assert await client.fairness.alist_audits() == SENTINEL

    def test_get_bias_history(self, client, mock_request):
        assert client.fairness.get_bias_history("m1") == SENTINEL

    async def test_aget_bias_history(self, client, mock_arequest):
        assert await client.fairness.aget_bias_history("m1") == SENTINEL

    def test_get_metric_history(self, client, mock_request):
        assert client.fairness.get_metric_history("m1", "demographic_parity") == SENTINEL

    async def test_aget_metric_history(self, client, mock_arequest):
        assert await client.fairness.aget_metric_history("m1", "demographic_parity") == SENTINEL

    def test_get_latest_audit(self, client, mock_request):
        assert client.fairness.get_latest_audit("m1") == SENTINEL

    async def test_aget_latest_audit(self, client, mock_arequest):
        assert await client.fairness.aget_latest_audit("m1") == SENTINEL


class TestAlertsResource:
    def test_create(self, client, mock_request):
        conditions = [{"metric_name": "accuracy", "operator": "lt", "threshold": 0.8}]
        result = client.alerts.create("a1", "drift", "high", conditions)
        assert result == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        # Regression: this used to hit /api/v1/alerts, which 404s -- the
        # real router mounts rule CRUD under /api/v1/alerts/rules.
        assert (method, endpoint) == ("POST", "/api/v1/alerts/rules")
        data = mock_request.call_args.kwargs["data"]
        assert data["severity"] == "high"
        assert data["conditions"] == conditions

    async def test_acreate(self, client, mock_arequest):
        conditions = [{"metric_name": "accuracy", "operator": "lt", "threshold": 0.8}]
        result = await client.alerts.acreate("a1", "drift", "high", conditions)
        assert result == SENTINEL

    def test_list(self, client, mock_request):
        assert client.alerts.list(model_id="m1") == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/alerts/rules",
            params={"model_id": "m1", "skip": 0, "limit": 100},
        )

    async def test_alist(self, client, mock_arequest):
        assert await client.alerts.alist() == SENTINEL

    def test_get_rule(self, client, mock_request):
        assert client.alerts.get_rule("r1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/alerts/rules/r1")

    async def test_aget_rule(self, client, mock_arequest):
        assert await client.alerts.aget_rule("r1") == SENTINEL

    def test_update_rule(self, client, mock_request):
        assert client.alerts.update_rule("r1", is_active=False) == SENTINEL
        mock_request.assert_called_once_with(
            "PATCH", "/api/v1/alerts/rules/r1", data={"is_active": False}
        )

    async def test_aupdate_rule(self, client, mock_arequest):
        assert await client.alerts.aupdate_rule("r1", is_active=False) == SENTINEL

    def test_delete_rule(self, client, mock_request):
        assert client.alerts.delete_rule("r1") == SENTINEL
        mock_request.assert_called_once_with("DELETE", "/api/v1/alerts/rules/r1")

    async def test_adelete_rule(self, client, mock_arequest):
        assert await client.alerts.adelete_rule("r1") == SENTINEL

    def test_evaluate_rule(self, client, mock_request):
        assert client.alerts.evaluate_rule("r1", metric_values={"accuracy": 0.5}) == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/alerts/rules/r1/evaluate")
        assert mock_request.call_args.kwargs["data"]["metric_values"] == {"accuracy": 0.5}

    async def test_aevaluate_rule(self, client, mock_arequest):
        assert await client.alerts.aevaluate_rule("r1") == SENTINEL

    def test_list_instances(self, client, mock_request):
        assert client.alerts.list_instances(severity="high") == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("GET", "/api/v1/alerts/instances")

    async def test_alist_instances(self, client, mock_arequest):
        assert await client.alerts.alist_instances() == SENTINEL

    def test_get_instance(self, client, mock_request):
        assert client.alerts.get_instance("a1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/alerts/instances/a1")

    async def test_aget_instance(self, client, mock_arequest):
        assert await client.alerts.aget_instance("a1") == SENTINEL

    def test_acknowledge(self, client, mock_request):
        assert client.alerts.acknowledge("a1", user_id="u1") == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/alerts/instances/a1/acknowledge")

    async def test_aacknowledge(self, client, mock_arequest):
        assert await client.alerts.aacknowledge("a1", user_id="u1") == SENTINEL

    def test_resolve(self, client, mock_request):
        assert client.alerts.resolve("a1", user_id="u1") == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/alerts/instances/a1/resolve")

    async def test_aresolve(self, client, mock_arequest):
        assert await client.alerts.aresolve("a1", user_id="u1") == SENTINEL

    def test_snooze(self, client, mock_request):
        assert client.alerts.snooze("a1", snooze_minutes=30) == SENTINEL
        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/alerts/instances/a1/snooze",
            data={"snooze_minutes": 30},
        )

    async def test_asnooze(self, client, mock_arequest):
        assert await client.alerts.asnooze("a1", snooze_minutes=30) == SENTINEL

    def test_statistics(self, client, mock_request):
        assert client.alerts.statistics(hours=48) == SENTINEL
        mock_request.assert_called_once_with(
            "GET", "/api/v1/alerts/statistics", params={"hours": 48}
        )

    async def test_astatistics(self, client, mock_arequest):
        assert await client.alerts.astatistics() == SENTINEL


class TestRiskRegisterResource:
    def test_list(self, client, mock_request):
        assert client.risk_register.list(status="identified", severity="high") == SENTINEL

    async def test_alist(self, client, mock_arequest):
        assert await client.risk_register.alist() == SENTINEL

    def test_get(self, client, mock_request):
        assert client.risk_register.get("r1") == SENTINEL

    async def test_aget(self, client, mock_arequest):
        assert await client.risk_register.aget("r1") == SENTINEL

    def test_portfolio(self, client, mock_request):
        assert client.risk_register.portfolio() == SENTINEL

    async def test_aportfolio(self, client, mock_arequest):
        assert await client.risk_register.aportfolio() == SENTINEL


class TestGovernanceResource:
    def test_list_boards(self, client, mock_request):
        assert client.governance.list_boards() == SENTINEL

    async def test_alist_boards(self, client, mock_arequest):
        assert await client.governance.alist_boards() == SENTINEL

    def test_list_review_requests(self, client, mock_request):
        result = client.governance.list_review_requests(board_id="b1", status="pending")
        assert result == SENTINEL

    async def test_alist_review_requests(self, client, mock_arequest):
        assert await client.governance.alist_review_requests() == SENTINEL

    def test_raci_grid(self, client, mock_request):
        assert client.governance.raci_grid(board_id="b1") == SENTINEL

    async def test_araci_grid(self, client, mock_arequest):
        assert await client.governance.araci_grid() == SENTINEL


class TestLLMResource:
    def test_log_call(self, client, mock_request):
        result = client.llm.log_call(
            provider="anthropic",
            model_name="claude-sonnet-5",
            prompt="hi",
            latency_ms=100.0,
            completion="hello",
            completion_tokens=5,
            prompt_tokens=1,
        )
        assert result == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/llm/logs")
        data = mock_request.call_args.kwargs["data"]
        assert data["provider"] == "anthropic"
        assert data["completion"] == "hello"
        # None-valued optional kwargs must not be sent at all
        assert "model_id" not in data

    async def test_alog_call(self, client, mock_arequest):
        assert (
            await client.llm.alog_call(
                provider="openai", model_name="gpt-4o", prompt="hi", latency_ms=1.0
            )
            == SENTINEL
        )

    def test_log_calls_batch(self, client, mock_request):
        logs = [
            {
                "provider": "openai",
                "model_name": "gpt-4o",
                "prompt": "hi",
                "latency_ms": 1.0,
            }
        ]
        assert client.llm.log_calls_batch(logs) == SENTINEL
        mock_request.assert_called_once_with("POST", "/api/v1/llm/logs/batch", data={"logs": logs})

    async def test_alog_calls_batch(self, client, mock_arequest):
        assert await client.llm.alog_calls_batch([]) == SENTINEL

    def test_get_stats(self, client, mock_request):
        assert client.llm.get_stats(model_id="m1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/llm/stats", params={"model_id": "m1"})

    async def test_aget_stats(self, client, mock_arequest):
        assert await client.llm.aget_stats() == SENTINEL

    def test_get_recent(self, client, mock_request):
        assert client.llm.get_recent(limit=10) == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/llm/logs/recent", params={"limit": 10})

    async def test_aget_recent(self, client, mock_arequest):
        assert await client.llm.aget_recent() == SENTINEL

    def test_get_log(self, client, mock_request):
        assert client.llm.get_log("l1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/llm/logs/l1")

    async def test_aget_log(self, client, mock_arequest):
        assert await client.llm.aget_log("l1") == SENTINEL

    def test_query_logs(self, client, mock_request):
        assert client.llm.query_logs(model_id="m1", limit=5) == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/llm/logs/query")
        data = mock_request.call_args.kwargs["data"]
        assert data == {"skip": 0, "limit": 5, "model_id": "m1"}

    async def test_aquery_logs(self, client, mock_arequest):
        assert await client.llm.aquery_logs() == SENTINEL

    def test_session_logs(self, client, mock_request):
        assert client.llm.session_logs("s1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/llm/logs/session/s1")

    async def test_asession_logs(self, client, mock_arequest):
        assert await client.llm.asession_logs("s1") == SENTINEL

    def test_cost_breakdown(self, client, mock_request):
        assert client.llm.cost_breakdown(include_user=True) == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/llm/costs/breakdown",
            params={"include_user": True, "include_environment": False},
        )

    async def test_acost_breakdown(self, client, mock_arequest):
        assert await client.llm.acost_breakdown() == SENTINEL

    def test_performance(self, client, mock_request):
        assert client.llm.performance(model_id="m1") == SENTINEL
        mock_request.assert_called_once_with(
            "GET", "/api/v1/llm/performance", params={"model_id": "m1"}
        )

    async def test_aperformance(self, client, mock_arequest):
        assert await client.llm.aperformance() == SENTINEL

    def test_trends_tokens(self, client, mock_request):
        assert client.llm.trends_tokens(granularity="week") == SENTINEL
        mock_request.assert_called_once_with(
            "GET", "/api/v1/llm/trends/tokens", params={"granularity": "week"}
        )

    async def test_atrends_tokens(self, client, mock_arequest):
        assert await client.llm.atrends_tokens() == SENTINEL

    def test_trends_costs(self, client, mock_request):
        assert client.llm.trends_costs(provider="openai") == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/llm/trends/costs",
            params={"granularity": "day", "provider": "openai"},
        )

    async def test_atrends_costs(self, client, mock_arequest):
        assert await client.llm.atrends_costs() == SENTINEL

    def test_usage_stats(self, client, mock_request):
        result = client.llm.usage_stats("2026-08-01", "2026-08-14")
        assert result == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/llm/usage-stats")

    async def test_ausage_stats(self, client, mock_arequest):
        assert await client.llm.ausage_stats("2026-08-01", "2026-08-14") == SENTINEL

    def test_cost_threshold_alert(self, client, mock_request):
        assert client.llm.cost_threshold_alert(threshold=50.0) == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/llm/alerts/cost-threshold",
            params={"period_minutes": 60, "threshold": 50.0},
        )

    async def test_acost_threshold_alert(self, client, mock_arequest):
        assert await client.llm.acost_threshold_alert() == SENTINEL

    def test_latency_threshold_alert(self, client, mock_request):
        assert client.llm.latency_threshold_alert() == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/llm/alerts/latency-threshold",
            params={"period_minutes": 60, "threshold_ms": 5000.0},
        )

    async def test_alatency_threshold_alert(self, client, mock_arequest):
        assert await client.llm.alatency_threshold_alert() == SENTINEL

    def test_error_rate_alert(self, client, mock_request):
        assert client.llm.error_rate_alert() == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/llm/alerts/error-rate",
            params={"period_minutes": 60, "threshold_percent": 10.0},
        )

    async def test_aerror_rate_alert(self, client, mock_arequest):
        assert await client.llm.aerror_rate_alert() == SENTINEL

    def test_cleanup_logs(self, client, mock_request):
        assert client.llm.cleanup_logs(days=7) == SENTINEL
        mock_request.assert_called_once_with(
            "DELETE", "/api/v1/llm/logs/cleanup", params={"days": 7}
        )

    async def test_acleanup_logs(self, client, mock_arequest):
        assert await client.llm.acleanup_logs() == SENTINEL


class TestRAGResource:
    def test_log_retrieval(self, client, mock_request):
        results = [{"document_id": "doc1", "rank": 1, "score": 0.9}]
        response = client.rag.log_retrieval(
            query="q",
            results=results,
            top_k=1,
            retrieval_method="vector",
            ground_truth_ids=["doc1"],
        )
        assert response == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/rag/retrievals")
        data = mock_request.call_args.kwargs["data"]
        assert data["query"] == "q"
        assert data["results"] == results
        assert data["ground_truth_ids"] == ["doc1"]
        assert "answer" not in data

    async def test_alog_retrieval(self, client, mock_arequest):
        assert (
            await client.rag.alog_retrieval(
                query="q", results=[], top_k=1, retrieval_method="vector"
            )
            == SENTINEL
        )

    def test_create_evaluation(self, client, mock_request):
        response = client.rag.create_evaluation(
            name="eval",
            queries=["q1", "q2"],
            ground_truth={"q1": ["doc1"]},
        )
        assert response == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/rag/evaluations")
        data = mock_request.call_args.kwargs["data"]
        assert data["queries"] == ["q1", "q2"]
        assert data["ground_truth"] == {"q1": ["doc1"]}

    async def test_acreate_evaluation(self, client, mock_arequest):
        assert await client.rag.acreate_evaluation(name="eval", queries=["q1"]) == SENTINEL

    def test_get_stats(self, client, mock_request):
        assert client.rag.get_stats(environment="prod") == SENTINEL
        mock_request.assert_called_once_with(
            "GET", "/api/v1/rag/stats", params={"environment": "prod"}
        )

    async def test_aget_stats(self, client, mock_arequest):
        assert await client.rag.aget_stats() == SENTINEL

    def test_list_evaluations(self, client, mock_request):
        assert client.rag.list_evaluations(limit=5) == SENTINEL
        mock_request.assert_called_once_with(
            "GET", "/api/v1/rag/evaluations", params={"limit": 5, "offset": 0}
        )

    async def test_alist_evaluations(self, client, mock_arequest):
        assert await client.rag.alist_evaluations() == SENTINEL

    def test_get_retrieval(self, client, mock_request):
        assert client.rag.get_retrieval("r1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/rag/retrievals/r1")

    async def test_aget_retrieval(self, client, mock_arequest):
        assert await client.rag.aget_retrieval("r1") == SENTINEL

    def test_list_retrievals(self, client, mock_request):
        assert client.rag.list_retrievals("m1", limit=10) == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/rag/retrievals",
            params={"model_id": "m1", "limit": 10, "offset": 0},
        )

    async def test_alist_retrievals(self, client, mock_arequest):
        assert await client.rag.alist_retrievals("m1") == SENTINEL

    def test_trends(self, client, mock_request):
        assert client.rag.trends(granularity="week") == SENTINEL
        mock_request.assert_called_once_with(
            "GET", "/api/v1/rag/trends", params={"granularity": "week"}
        )

    async def test_atrends(self, client, mock_arequest):
        assert await client.rag.atrends() == SENTINEL

    def test_get_evaluation(self, client, mock_request):
        assert client.rag.get_evaluation("ev1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/rag/evaluations/ev1")

    async def test_aget_evaluation(self, client, mock_arequest):
        assert await client.rag.aget_evaluation("ev1") == SENTINEL

    def test_metrics_precision(self, client, mock_request):
        assert client.rag.metrics_precision(model_id="m1") == SENTINEL
        mock_request.assert_called_once_with(
            "GET", "/api/v1/rag/metrics/precision", params={"model_id": "m1"}
        )

    async def test_ametrics_precision(self, client, mock_arequest):
        assert await client.rag.ametrics_precision() == SENTINEL

    def test_metrics_relevance(self, client, mock_request):
        assert client.rag.metrics_relevance(model_id="m1") == SENTINEL
        mock_request.assert_called_once_with(
            "GET", "/api/v1/rag/metrics/relevance", params={"model_id": "m1"}
        )

    async def test_ametrics_relevance(self, client, mock_arequest):
        assert await client.rag.ametrics_relevance() == SENTINEL


class TestSafetyResource:
    def test_analyze(self, client, mock_request):
        response = client.safety.analyze("some content", model_id="m1")
        assert response == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/safety/analyze")
        data = mock_request.call_args.kwargs["data"]
        assert data["content"] == "some content"
        assert data["model_id"] == "m1"
        assert data["check_toxicity"] is True

    async def test_aanalyze(self, client, mock_arequest):
        assert await client.safety.aanalyze("content") == SENTINEL

    def test_analyze_batch(self, client, mock_request):
        response = client.safety.analyze_batch(["a", "b"], environment="staging")
        assert response == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/safety/analyze/batch")
        data = mock_request.call_args.kwargs["data"]
        assert data["contents"] == ["a", "b"]
        assert data["environment"] == "staging"

    async def test_aanalyze_batch(self, client, mock_arequest):
        assert await client.safety.aanalyze_batch(["a"]) == SENTINEL

    def test_get_scores(self, client, mock_request):
        assert client.safety.get_scores(contains_pii=True) == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/safety/scores",
            params={"contains_pii": True, "limit": 100, "offset": 0},
        )

    async def test_aget_scores(self, client, mock_arequest):
        assert await client.safety.aget_scores() == SENTINEL

    def test_get_stats(self, client, mock_request):
        assert client.safety.get_stats(model_id="m1") == SENTINEL
        mock_request.assert_called_once_with(
            "GET", "/api/v1/safety/stats", params={"model_id": "m1"}
        )

    async def test_aget_stats(self, client, mock_arequest):
        assert await client.safety.aget_stats() == SENTINEL

    def test_get_score(self, client, mock_request):
        assert client.safety.get_score("s1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/safety/scores/s1")

    async def test_aget_score(self, client, mock_arequest):
        assert await client.safety.aget_score("s1") == SENTINEL

    def test_trends(self, client, mock_request):
        assert client.safety.trends(granularity="week") == SENTINEL
        mock_request.assert_called_once_with(
            "GET", "/api/v1/safety/trends", params={"granularity": "week"}
        )

    async def test_atrends(self, client, mock_arequest):
        assert await client.safety.atrends() == SENTINEL

    def test_create_threshold(self, client, mock_request):
        assert client.safety.create_threshold("strict", block_pii=True) == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/safety/thresholds")
        data = mock_request.call_args.kwargs["data"]
        assert data == {"name": "strict", "block_pii": True}

    async def test_acreate_threshold(self, client, mock_arequest):
        assert await client.safety.acreate_threshold("strict") == SENTINEL

    def test_list_thresholds(self, client, mock_request):
        assert client.safety.list_thresholds(is_active=True) == SENTINEL
        mock_request.assert_called_once_with(
            "GET", "/api/v1/safety/thresholds", params={"is_active": True}
        )

    async def test_alist_thresholds(self, client, mock_arequest):
        assert await client.safety.alist_thresholds() == SENTINEL

    def test_get_threshold(self, client, mock_request):
        assert client.safety.get_threshold("t1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/safety/thresholds/t1")

    async def test_aget_threshold(self, client, mock_arequest):
        assert await client.safety.aget_threshold("t1") == SENTINEL

    def test_update_threshold(self, client, mock_request):
        assert client.safety.update_threshold("t1", is_active=False) == SENTINEL
        mock_request.assert_called_once_with(
            "PATCH", "/api/v1/safety/thresholds/t1", data={"is_active": False}
        )

    async def test_aupdate_threshold(self, client, mock_arequest):
        assert await client.safety.aupdate_threshold("t1", is_active=False) == SENTINEL

    def test_delete_threshold(self, client, mock_request):
        assert client.safety.delete_threshold("t1") == SENTINEL
        mock_request.assert_called_once_with("DELETE", "/api/v1/safety/thresholds/t1")

    async def test_adelete_threshold(self, client, mock_arequest):
        assert await client.safety.adelete_threshold("t1") == SENTINEL


class TestLLMXAIResource:
    def test_attention(self, client, mock_request):
        assert client.llm_xai.attention("hi", attention_weights=[[0.1]]) == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/llm-xai/attention")
        data = mock_request.call_args.kwargs["data"]
        assert data["prompt"] == "hi"
        assert data["attention_weights"] == [[0.1]]

    async def test_aattention(self, client, mock_arequest):
        assert await client.llm_xai.aattention("hi") == SENTINEL

    def test_token_importance(self, client, mock_request):
        assert client.llm_xai.token_importance("hi", "there") == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/llm-xai/token-importance")

    async def test_atoken_importance(self, client, mock_arequest):
        assert await client.llm_xai.atoken_importance("hi", "there") == SENTINEL

    def test_prompt_sensitivity(self, client, mock_request):
        assert client.llm_xai.prompt_sensitivity("hi", "there") == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/llm-xai/sensitivity")

    async def test_aprompt_sensitivity(self, client, mock_arequest):
        assert await client.llm_xai.aprompt_sensitivity("hi", "there") == SENTINEL

    def test_counterfactuals(self, client, mock_request):
        assert client.llm_xai.counterfactuals("hi", "there") == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/llm-xai/counterfactuals")

    async def test_acounterfactuals(self, client, mock_arequest):
        assert await client.llm_xai.acounterfactuals("hi", "there") == SENTINEL

    def test_debug_prompt(self, client, mock_request):
        assert client.llm_xai.debug_prompt("hi") == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/llm-xai/debug-prompt")

    async def test_adebug_prompt(self, client, mock_arequest):
        assert await client.llm_xai.adebug_prompt("hi") == SENTINEL

    def test_get_explanation(self, client, mock_request):
        assert client.llm_xai.get_explanation("e1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/llm-xai/explanations/e1")

    async def test_aget_explanation(self, client, mock_arequest):
        assert await client.llm_xai.aget_explanation("e1") == SENTINEL

    def test_get_prompt_analysis(self, client, mock_request):
        assert client.llm_xai.get_prompt_analysis("a1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/llm-xai/prompt-analyses/a1")

    async def test_aget_prompt_analysis(self, client, mock_arequest):
        assert await client.llm_xai.aget_prompt_analysis("a1") == SENTINEL

    def test_list_explanations(self, client, mock_request):
        assert client.llm_xai.list_explanations(model_id="m1") == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/llm-xai/explanations",
            params={"model_id": "m1", "limit": 50},
        )

    async def test_alist_explanations(self, client, mock_arequest):
        assert await client.llm_xai.alist_explanations() == SENTINEL

    def test_list_prompt_analyses(self, client, mock_request):
        assert client.llm_xai.list_prompt_analyses(model_id="m1") == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/llm-xai/prompt-analyses",
            params={"model_id": "m1", "limit": 50},
        )

    async def test_alist_prompt_analyses(self, client, mock_arequest):
        assert await client.llm_xai.alist_prompt_analyses() == SENTINEL

    def test_stats(self, client, mock_request):
        assert client.llm_xai.stats(model_id="m1") == SENTINEL
        mock_request.assert_called_once_with(
            "GET", "/api/v1/llm-xai/stats", params={"model_id": "m1", "days": 7}
        )

    async def test_astats(self, client, mock_arequest):
        assert await client.llm_xai.astats() == SENTINEL

    def test_visualize_attention(self, client, mock_request):
        assert client.llm_xai.visualize_attention("e1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/llm-xai/attention/visualize/e1")

    async def test_avisualize_attention(self, client, mock_arequest):
        assert await client.llm_xai.avisualize_attention("e1") == SENTINEL

    def test_visualize_token_importance(self, client, mock_request):
        assert client.llm_xai.visualize_token_importance("e1", top_k=5) == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/llm-xai/token-importance/visualize/e1",
            params={"top_k": 5},
        )

    async def test_avisualize_token_importance(self, client, mock_arequest):
        assert await client.llm_xai.avisualize_token_importance("e1") == SENTINEL

    def test_batch_analyze(self, client, mock_request):
        assert client.llm_xai.batch_analyze("l1") == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/llm-xai/batch-analyze")
        params = mock_request.call_args.kwargs["params"]
        assert params["llm_log_id"] == "l1"

    async def test_abatch_analyze(self, client, mock_arequest):
        assert await client.llm_xai.abatch_analyze("l1") == SENTINEL


class TestAgentWorkflowsResource:
    def test_create_and_start(self, client, mock_request):
        assert client.agent_workflows.create_and_start("wf1", "crewai") == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/workflows/multi-agent/start")
        data = mock_request.call_args.kwargs["data"]
        assert data == {"name": "wf1", "framework": "crewai"}

    async def test_acreate_and_start(self, client, mock_arequest):
        assert await client.agent_workflows.acreate_and_start("wf1", "crewai") == SENTINEL

    def test_start(self, client, mock_request):
        assert client.agent_workflows.start("w1", inputs={"x": 1}) == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/workflows/multi-agent/w1/start")

    async def test_astart(self, client, mock_arequest):
        assert await client.agent_workflows.astart("w1") == SENTINEL

    def test_complete(self, client, mock_request):
        assert client.agent_workflows.complete("w1", "completed") == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == (
            "POST",
            "/api/v1/workflows/multi-agent/w1/complete",
        )
        assert mock_request.call_args.kwargs["params"] == {"trigger_analytics": True}

    async def test_acomplete(self, client, mock_arequest):
        assert await client.agent_workflows.acomplete("w1", "completed") == SENTINEL

    def test_list(self, client, mock_request):
        assert client.agent_workflows.list(framework="crewai") == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/workflows/multi-agent",
            params={"skip": 0, "limit": 100, "framework": "crewai"},
        )

    async def test_alist(self, client, mock_arequest):
        assert await client.agent_workflows.alist() == SENTINEL

    def test_get(self, client, mock_request):
        assert client.agent_workflows.get("w1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/workflows/multi-agent/w1")

    async def test_aget(self, client, mock_arequest):
        assert await client.agent_workflows.aget("w1") == SENTINEL

    def test_delete(self, client, mock_request):
        assert client.agent_workflows.delete("w1") == SENTINEL
        mock_request.assert_called_once_with("DELETE", "/api/v1/workflows/multi-agent/w1")

    async def test_adelete(self, client, mock_arequest):
        assert await client.agent_workflows.adelete("w1") == SENTINEL

    def test_register_agent(self, client, mock_request):
        assert client.agent_workflows.register_agent("w1", "Researcher") == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == (
            "POST",
            "/api/v1/workflows/multi-agent/w1/agents",
        )
        assert mock_request.call_args.kwargs["data"] == {"name": "Researcher"}

    async def test_aregister_agent(self, client, mock_arequest):
        assert await client.agent_workflows.aregister_agent("w1", "Researcher") == SENTINEL

    def test_list_agents(self, client, mock_request):
        assert client.agent_workflows.list_agents("w1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/workflows/multi-agent/w1/agents")

    async def test_alist_agents(self, client, mock_arequest):
        assert await client.agent_workflows.alist_agents("w1") == SENTINEL

    def test_create_execution(self, client, mock_request):
        assert client.agent_workflows.create_execution("w1", "ag1") == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == (
            "POST",
            "/api/v1/workflows/multi-agent/w1/executions",
        )
        assert mock_request.call_args.kwargs["data"] == {"agent_id": "ag1"}

    async def test_acreate_execution(self, client, mock_arequest):
        assert await client.agent_workflows.acreate_execution("w1", "ag1") == SENTINEL

    def test_log_interaction(self, client, mock_request):
        assert client.agent_workflows.log_interaction("w1", "delegation") == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == (
            "POST",
            "/api/v1/workflows/multi-agent/w1/interactions",
        )

    async def test_alog_interaction(self, client, mock_arequest):
        assert await client.agent_workflows.alog_interaction("w1", "delegation") == SENTINEL

    def test_list_interactions(self, client, mock_request):
        assert client.agent_workflows.list_interactions("w1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/workflows/multi-agent/w1/interactions")

    async def test_alist_interactions(self, client, mock_arequest):
        assert await client.agent_workflows.alist_interactions("w1") == SENTINEL

    def test_create_task(self, client, mock_request):
        assert client.agent_workflows.create_task("w1", "Research") == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == (
            "POST",
            "/api/v1/workflows/multi-agent/w1/tasks",
        )
        data = mock_request.call_args.kwargs["data"]
        assert data == {"task_name": "Research", "priority": 0}

    async def test_acreate_task(self, client, mock_arequest):
        assert await client.agent_workflows.acreate_task("w1", "Research") == SENTINEL

    def test_update_task_status(self, client, mock_request):
        assert client.agent_workflows.update_task_status("t1", "completed") == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == (
            "PATCH",
            "/api/v1/workflows/multi-agent/tasks/t1",
        )

    async def test_aupdate_task_status(self, client, mock_arequest):
        assert await client.agent_workflows.aupdate_task_status("t1", "completed") == SENTINEL

    def test_list_tasks(self, client, mock_request):
        assert client.agent_workflows.list_tasks("w1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/workflows/multi-agent/w1/tasks")

    async def test_alist_tasks(self, client, mock_arequest):
        assert await client.agent_workflows.alist_tasks("w1") == SENTINEL

    def test_analytics(self, client, mock_request):
        assert client.agent_workflows.analytics("w1") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/workflows/multi-agent/w1/analytics")

    async def test_aanalytics(self, client, mock_arequest):
        assert await client.agent_workflows.aanalytics("w1") == SENTINEL

    def test_cost_breakdown_uses_extended_timeout(self, client, mock_request):
        """Regression: the backend blocks server-side up to 30s via Celery
        .get(timeout=30) for this endpoint, so the client timeout must be
        strictly greater than that ceiling."""
        assert client.agent_workflows.cost_breakdown("w1") == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/workflows/multi-agent/w1/cost-breakdown",
            timeout=40,
        )

    async def test_acost_breakdown_uses_extended_timeout(self, client, mock_arequest):
        assert await client.agent_workflows.acost_breakdown("w1") == SENTINEL
        mock_arequest.assert_called_once_with(
            "GET",
            "/api/v1/workflows/multi-agent/w1/cost-breakdown",
            timeout=40,
        )

    def test_bottlenecks_uses_extended_timeout(self, client, mock_request):
        assert client.agent_workflows.bottlenecks("w1") == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/workflows/multi-agent/w1/bottlenecks",
            timeout=40,
        )

    async def test_abottlenecks_uses_extended_timeout(self, client, mock_arequest):
        assert await client.agent_workflows.abottlenecks("w1") == SENTINEL
        mock_arequest.assert_called_once_with(
            "GET",
            "/api/v1/workflows/multi-agent/w1/bottlenecks",
            timeout=40,
        )

    def test_timeline_uses_extended_timeout(self, client, mock_request):
        assert client.agent_workflows.timeline("w1") == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/workflows/multi-agent/w1/timeline",
            timeout=40,
        )

    async def test_atimeline_uses_extended_timeout(self, client, mock_arequest):
        assert await client.agent_workflows.atimeline("w1") == SENTINEL
        mock_arequest.assert_called_once_with(
            "GET",
            "/api/v1/workflows/multi-agent/w1/timeline",
            timeout=40,
        )


class TestMetricsResource:
    def test_create(self, client, mock_request):
        assert client.metrics.create("m1", "accuracy", "accuracy", 0.9) == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == ("POST", "/api/v1/metrics/")
        data = mock_request.call_args.kwargs["data"]
        assert data == {
            "model_id": "m1",
            "metric_type": "accuracy",
            "metric_name": "accuracy",
            "metric_value": 0.9,
        }

    async def test_acreate(self, client, mock_arequest):
        assert await client.metrics.acreate("m1", "accuracy", "accuracy", 0.9) == SENTINEL

    def test_create_batch(self, client, mock_request):
        metrics_payload = [{"metric_type": "accuracy", "metric_value": 0.9}]
        assert client.metrics.create_batch("m1", metrics_payload) == SENTINEL
        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/metrics/batch",
            data={"model_id": "m1", "metrics": metrics_payload},
        )

    async def test_acreate_batch(self, client, mock_arequest):
        assert await client.metrics.acreate_batch("m1", []) == SENTINEL

    def test_calculate_classification(self, client, mock_request):
        assert client.metrics.calculate_classification([1], [1]) == SENTINEL
        method, endpoint = mock_request.call_args[0][:2]
        assert (method, endpoint) == (
            "POST",
            "/api/v1/metrics/calculate/classification",
        )
        assert mock_request.call_args.kwargs["params"] == {
            "average": "binary",
            "pos_label": 1,
        }

    async def test_acalculate_classification(self, client, mock_arequest):
        assert await client.metrics.acalculate_classification([1], [1]) == SENTINEL

    def test_calculate_regression(self, client, mock_request):
        assert client.metrics.calculate_regression([1.0], [1.1]) == SENTINEL
        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/metrics/calculate/regression",
            data={"y_true": [1.0], "y_pred": [1.1]},
        )

    async def test_acalculate_regression(self, client, mock_arequest):
        assert await client.metrics.acalculate_regression([1.0], [1.1]) == SENTINEL

    def test_get_model_metrics(self, client, mock_request):
        assert client.metrics.get_model_metrics("m1") == SENTINEL
        mock_request.assert_called_once_with(
            "GET", "/api/v1/metrics/m1", params={"skip": 0, "limit": 100}
        )

    async def test_aget_model_metrics(self, client, mock_arequest):
        assert await client.metrics.aget_model_metrics("m1") == SENTINEL

    def test_latest(self, client, mock_request):
        assert client.metrics.latest("m1", "accuracy") == SENTINEL
        mock_request.assert_called_once_with("GET", "/api/v1/metrics/m1/latest/accuracy")

    async def test_alatest(self, client, mock_arequest):
        assert await client.metrics.alatest("m1", "accuracy") == SENTINEL

    def test_timeseries(self, client, mock_request):
        assert client.metrics.timeseries("m1", "accuracy", "2026-08-01", "2026-08-14") == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/metrics/m1/timeseries/accuracy",
            params={"start_date": "2026-08-01", "end_date": "2026-08-14"},
        )

    async def test_atimeseries(self, client, mock_arequest):
        assert (
            await client.metrics.atimeseries("m1", "accuracy", "2026-08-01", "2026-08-14")
            == SENTINEL
        )

    def test_aggregate(self, client, mock_request):
        assert client.metrics.aggregate("m1", "daily") == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/metrics/m1/aggregate",
            params={"period": "daily", "force_recompute": False},
        )

    async def test_aaggregate(self, client, mock_arequest):
        assert await client.metrics.aaggregate("m1", "daily") == SENTINEL

    def test_trend(self, client, mock_request):
        assert client.metrics.trend("m1", "accuracy") == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/metrics/m1/trend/accuracy",
            params={"lookback_days": 30},
        )

    async def test_atrend(self, client, mock_arequest):
        assert await client.metrics.atrend("m1", "accuracy") == SENTINEL

    def test_rolling(self, client, mock_request):
        assert client.metrics.rolling("m1", "accuracy") == SENTINEL
        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/metrics/m1/rolling/accuracy",
            params={"window_days": 7},
        )

    async def test_arolling(self, client, mock_arequest):
        assert await client.metrics.arolling("m1", "accuracy") == SENTINEL

    def test_summary(self, client, mock_request):
        assert client.metrics.summary("m1") == SENTINEL
        mock_request.assert_called_once_with(
            "GET", "/api/v1/metrics/m1/summary", params={"days": 30}
        )

    async def test_asummary(self, client, mock_arequest):
        assert await client.metrics.asummary("m1") == SENTINEL

    def test_delete(self, client, mock_request):
        assert client.metrics.delete("m1") == SENTINEL
        mock_request.assert_called_once_with("DELETE", "/api/v1/metrics/m1")

    async def test_adelete(self, client, mock_arequest):
        assert await client.metrics.adelete("m1") == SENTINEL


class TestRequestOrQueueFallback:
    """_request_or_queue() (used by register/update_baseline/log/log_batch)
    falls back to the offline queue on APIConnectionError when offline mode
    is enabled, and re-raises when it isn't."""

    def test_reraises_when_offline_disabled(self, client):
        from whiteboxxai.exceptions import APIConnectionError

        with patch.object(client, "request", side_effect=APIConnectionError("down")):
            with patch.object(client, "is_offline_enabled", return_value=False):
                with pytest.raises(APIConnectionError):
                    client.models.register("m", "classification")

    def test_queues_when_offline_enabled(self, client):
        from whiteboxxai.exceptions import APIConnectionError

        with patch.object(client, "request", side_effect=APIConnectionError("down")):
            with patch.object(client, "is_offline_enabled", return_value=True):
                with patch.object(client, "_offline_manager") as mock_manager:
                    mock_manager.queue.enqueue.return_value = "op-123"
                    result = client.models.register("m", "classification")

        assert result == {"status": "queued", "operation_id": "op-123"}
