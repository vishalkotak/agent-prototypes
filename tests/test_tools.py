import json

from triage_agent.tools import TOOL_REGISTRY, execute_tool


VALID_ARGS = {
    "service": "checkout-api",
    "metric": "error_rate",
    "window_minutes": 15,
}


def error_payload(result):
    """Validation errors are serialized as a JSON string for the model to read."""
    return json.loads(result.error)


def test_registry_exposes_query_metrics_as_read_only():
    tool = TOOL_REGISTRY["query_metrics"]
    assert tool.name == "query_metrics"
    assert tool.read_only is True


def test_valid_call_echoes_the_query_and_returns_values():
    result = execute_tool("query_metrics", VALID_ARGS)

    assert result.success is True
    assert result.error is None
    assert result.content["service"] == "checkout-api"
    assert result.content["metric"] == "error_rate"
    assert result.content["window_minutes"] == 15
    assert result.content["values"]


def test_unknown_tool_is_reported_not_raised():
    result = execute_tool("delete_production", VALID_ARGS)

    assert result.success is False
    assert result.content is None
    assert "delete_production" in result.error


def test_unknown_metric_is_rejected_with_the_allowed_values():
    result = execute_tool("query_metrics", {**VALID_ARGS, "metric": "disk_io"})

    assert result.success is False
    # The enum members are what let the model retry with a legal value.
    assert "latency_ms" in result.error


def test_out_of_range_window_is_rejected():
    result = execute_tool("query_metrics", {**VALID_ARGS, "window_minutes": 1440})

    assert result.success is False
    assert error_payload(result)[0]["loc"] == ["window_minutes"]


def test_empty_service_is_rejected():
    result = execute_tool("query_metrics", {**VALID_ARGS, "service": ""})

    assert result.success is False
    assert error_payload(result)[0]["loc"] == ["service"]


def test_missing_arguments_are_all_reported_at_once():
    result = execute_tool("query_metrics", {})

    assert result.success is False
    reported = {entry["loc"][0] for entry in error_payload(result)}
    assert reported == {"service", "metric", "window_minutes"}


def test_validation_errors_omit_input_echo_and_doc_urls():
    """Both are prompt-token waste: the model already knows what it sent."""
    result = execute_tool("query_metrics", {**VALID_ARGS, "service": ""})

    for entry in error_payload(result):
        assert "input" not in entry
        assert "url" not in entry
