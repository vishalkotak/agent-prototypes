from dataclasses import dataclass
from typing import Any, Callable
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from collections import Counter

tool_attempts: Counter[str] = Counter()

@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    arguments_model: type[BaseModel]
    handler: Callable[[BaseModel], Any]
    read_only: bool

@dataclass(frozen=True)
class ToolResult:
    success: bool
    content: Any = None
    error_code: str | None = None
    error: str | None = None

@dataclass
class RunState:
    model_steps: int = 0
    # Tracking tool requests because what if the model is making the same incorrect 
    # tool requests repeatedly.
    tool_requests: int = 0
    tool_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0

class ToolArgs(BaseModel):
    """Base for every tool's arguments.

    ``extra="forbid"`` means an argument the model invented is a validation
    error rather than a silently dropped key, so the model finds out it
    guessed wrong. It also makes ``model_json_schema()`` emit
    ``additionalProperties: false``, which the Messages API requires before
    it will accept a tool definition as ``strict``.
    """

    model_config = ConfigDict(extra="forbid")

class QueryMetricsArgs(ToolArgs):
    service: str = Field(
        min_length=1,
        description="Name of the service to investigate",
    )
    metric: Literal[
        "latency_ms",
        "error_rate",
        "request_rate",
        "cpu_percent",
    ]
    window_minutes: int = Field(
        ge=1,
        le=60,
        description="Number of recent minutes to query",
    )

METRIC_UNITS = {
    "latency_ms": "milliseconds",
    "error_rate": "percent",
    "request_rate": "requests_per_second",
    "cpu_percent": "percent",
}

def query_metrics(args: QueryMetricsArgs) -> dict[str, Any]:
    return {
        "service": args.service,
        "metric": args.metric,
        "window_minutes": args.window_minutes,
        "values": [0.2, 0.3, 14.1],
        "unit": METRIC_UNITS[args.metric],
    }

TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "query_metrics": ToolDefinition(
        name="query_metrics",
        description="Query a metric for a service over a recent time window.",
        arguments_model=QueryMetricsArgs,
        handler=query_metrics,
        read_only=True,
    )
}

def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.arguments_model.model_json_schema(),
            "strict": True,
        }
        for tool in TOOL_REGISTRY.values()
    ]

def describe_handler_failure(tool_name: str, exception: Exception) -> str:
    return f"{tool_name} failed: {type(exception).__name__}: {exception}"

def execute_tool(
    tool_name: str,
    raw_arguments: dict[str, Any],
) -> ToolResult:
    tool_definition = TOOL_REGISTRY.get(tool_name)
    if tool_definition is None:
        return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
    try:
        args_after_validation = tool_definition.arguments_model.model_validate(raw_arguments)
    except ValidationError as e:
        return ToolResult(
            success=False,
            error=e.json(
                include_input=False,
                include_url=False,
            ),
        )
    try:
        result = tool_definition.handler(args_after_validation)
    except TimeoutError:
        return ToolResult(
            success=False,
            error_code="tool_timeout",
            error=f"{tool_name} timed out",
        )
    return ToolResult(success=True, content=result)
