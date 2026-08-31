from triage_agent.tools import QueryMetricsArgs, query_metrics, search_logs, SearchLogsArgs


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

def test_search_logs_filters_by_query_and_window() -> None:
    result = search_logs(
        SearchLogsArgs(
            service="checkout",
            query="timeout",
            window_minutes=10,
        )
    )

    assert result["truncated"] is False
    assert len(result["matches"]) == 2

    assert all(
        event["minutes_ago"] <= 10
        for event in result["matches"]
    )

    assert all(
        "timeout" in event["message"].casefold()
        for event in result["matches"]
    )