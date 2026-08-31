import logging
from uuid import uuid4
from collections.abc import Sequence
from typing import Any, Protocol

from triage_agent.tools import execute_tool, ToolResult

from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

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
    def decide(self, history: list[Any], *, force_final: bool = False,) -> ModelDecision:
        ...

class ScriptedModel:
    def __init__(self, decisions: Sequence[ModelDecision]) -> None:
        self._decisions = iter(decisions)
        self.received_histories: list[list[Any]] = []

    def decide(self, history: list[Any], *, force_final: bool = False,) -> ModelDecision:
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
    run_id = uuid4().hex[:8]
    history: list[Any] = [
        {
            "role": "user",
            "content": user_task,
        }
    ]
    logger.info(
        "event=agent_run_started run_id=%s max_model_steps=%d",
        run_id,
        limits.max_model_steps,
    )
    for step in range(1, limits.max_model_steps + 1):
        force_final = step == limits.max_model_steps
        logger.info(
            "event=model_step_started "
            "run_id=%s step=%d force_final=%s",
            run_id,
            step,
            force_final,
        )
        decision = model.decide(history, force_final=force_final)
        if isinstance(decision, FinalAnswer):
            logger.info(
                "event=agent_run_completed "
                "run_id=%s step=%d history_events=%d",
                run_id,
                step,
                len(history),
            )
            return decision
        logger.info(
            "event=tool_requested "
            "run_id=%s step=%d call_id=%s tool=%s",
            run_id,
            step,
            decision.call_id,
            decision.tool_name,
        )
        history.append(decision)
        tool_result = execute_tool(
            tool_name=decision.tool_name,
            raw_arguments=decision.arguments,
        )
        logger.info(
            "event=tool_completed "
            "run_id=%s call_id=%s tool=%s success=%s "
            "error_code=%s",
            run_id,
            decision.call_id,
            decision.tool_name,
            tool_result.success,
            tool_result.error_code,
        )
        history.append(
            ToolCallResult(
                call_id=decision.call_id,
                result=tool_result,
            )
        )
        if force_final and isinstance(decision, ToolCall):
            logger.error(
                "event=model_violated_force_final "
                "run_id=%s step=%d",
                run_id,
                step,
            )

            raise RuntimeError(
                "Model requested a tool when a final answer was required"
            )
    logger.error(
        "event=agent_step_limit_exceeded "
        "run_id=%s max_model_steps=%d history_events=%d",
        run_id,
        limits.max_model_steps,
        len(history),
    )
    raise RuntimeError("Agent exceeded maximum model steps")