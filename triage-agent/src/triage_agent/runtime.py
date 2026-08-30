from collections.abc import Sequence
from typing import Any, Protocol

from triage_agent.tools import execute_tool, ToolResult

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]

@dataclass(frozen=True)
class ToolCallResult:
    call_id: str
    result: ToolResult

@dataclass(frozen=True)
class FinalAnswer:
    content: str

@dataclass(frozen=True)
class RunLimits:
    max_model_steps: int = 8

ModelDecision = ToolCall | FinalAnswer

class ModelClient(Protocol):
    def decide(self, history: list[Any]) -> ModelDecision:
        ...

class ScriptedModel:
    def __init__(self, decisions: Sequence[ModelDecision]) -> None:
        self._decisions = iter(decisions)
        self.received_histories: list[list[Any]] = []

    def decide(self, history: list[Any]) -> ModelDecision:
        self.received_histories.append(history.copy())
        return next(self._decisions)

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

def run_agent(
    model: ModelClient,
    user_task: str,
    limits: RunLimits,
) -> FinalAnswer:
    history: list[Any] = [
        {
            "role": "user",
            "content": user_task,
        }
    ]
    for step in range(limits.max_model_steps):
        decision = model.decide(history)
        print(
            f"step={step + 1} "
            f"decision={decision!r}"
        )
        if isinstance(decision, FinalAnswer):
            return decision
        history.append(decision)
        tool_result = execute_tool(
            tool_name=decision.tool_name,
            raw_arguments=decision.arguments,
        )
        history.append(
            ToolCallResult(
                call_id=decision.call_id,
                result=tool_result,
            )
        )
    raise RuntimeError("Agent exceeded maximum model steps")