from triage_agent.tools import QueryMetricsArgs, query_metrics


def test_query_metrics_respects_window_and_unit() -> None:
    result = query_metrics(
        QueryMetricsArgs(
            service="checkout",
            metric="error_rate",
            window_minutes=10,
        )
    )

    assert result["unit"] == "percent"
    assert result["samples"] == [
        {"minutes_ago": 10, "value": 0.3},
        {"minutes_ago": 2, "value": 14.1},
    ]