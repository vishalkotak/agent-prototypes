from typing import Any

from triage_agent.runtime import (
    FinalAnswer,
    RunLimits,
    ScriptedModel,
    ToolCall,
    run_agent,
    ToolCallResult,
    ModelDecision,
)

def test_agent_executes_tool_and_returns_final_answer() -> None:
    model = ScriptedModel(
        decisions=[
            ToolCall(
                call_id="call_1",
                tool_name="query_metrics",
                arguments={
                    "service": "checkout",
                    "metric": "error_rate",
                    "window_minutes": 30,
                },
            ),
            FinalAnswer(
                content="Checkout error rate increased sharply."
            ),
        ]
    )

    answer = run_agent(
        model=model,
        user_task="Investigate checkout failures",
        limits=RunLimits(max_model_steps=2),
    )

    assert answer.content == "Checkout error rate increased sharply."
    assert len(model.received_histories) == 2

    # On the second call, the model received:
    # user task → tool call → tool result
    second_history = model.received_histories[1]
    assert len(second_history) == 3
    assert isinstance(second_history[1], ToolCall)
    assert isinstance(second_history[2], ToolCallResult)

    tool_call = second_history[1]
    tool_call_result = second_history[2]
    assert tool_call.call_id == "call_1"
    assert tool_call_result.call_id == "call_1"
    assert tool_call_result.result.success is True
    assert tool_call_result.result.content["service"] == "checkout"
    assert tool_call_result.result.content["metric"] == "error_rate"


def test_agent_forces_final_decision_on_last_step() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.force_final_values: list[bool] = []

        def decide(
            self,
            history: list[Any],
            *,
            force_final: bool = False,
        ) -> ModelDecision:
            self.force_final_values.append(force_final)

            if force_final:
                return FinalAnswer(content="Best available diagnosis")

            return ToolCall(
                call_id="call_1",
                tool_name="query_metrics",
                arguments={
                    "service": "checkout",
                    "metric": "error_rate",
                    "window_minutes": 30,
                },
            )

    model = RecordingModel()

    answer = run_agent(
        model=model,
        user_task="Investigate checkout errors",
        limits=RunLimits(max_model_steps=2),
    )

    assert answer.content == "Best available diagnosis"
    assert model.force_final_values == [False, True]