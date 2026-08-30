from dataclasses import dataclass
from typing import Any, Callable
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

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
    error: str | None = None

class QueryMetricsArgs(BaseModel):
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

def query_metrics(args: QueryMetricsArgs) -> dict[str, Any]:
    return {
        "service": args.service,
        "metric": args.metric,
        "window_minutes": args.window_minutes,
        "values": [0.2, 0.3, 14.1],
        "unit": "percent",
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

def execute_tool(
    tool_name: str,
    raw_arguments: dict[str, Any],
) -> ToolResult:
    tool_definition = TOOL_REGISTRY.get(tool_name)
    if tool_definition is None:
        return ToolResult(False,  error=f"Unknown tool: {tool_name}")
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
    result = tool_definition.handler(args_after_validation)
    return ToolResult(True, content=result)

