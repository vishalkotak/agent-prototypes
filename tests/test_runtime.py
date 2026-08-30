from triage_agent.runtime import (
    FinalAnswer,
    RunLimits,
    ScriptedModel,
    ToolCall,
    run_agent,
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