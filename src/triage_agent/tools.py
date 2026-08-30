import logging
from dataclasses import dataclass
from typing import Any, Callable
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

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
    """Render the registry as the Messages API ``tools`` payload.

    Every arguments model derives from :class:`ToolArgs`, so each schema
    carries ``additionalProperties: false`` and can be declared ``strict``:
    the API then guarantees ``tool_use.input`` validates against the schema
    before it ever reaches ``execute_tool``.
    """
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
    """Turn a handler crash into a line the model is allowed to read.

    This is the trust boundary: whatever this returns is sent back to the
    model as tool output. Naming the exception type and message helps the
    model decide whether to retry, try another tool, or give up -- at the
    cost of exposing whatever a backend client happened to put in the
    message. Tighten this if handlers start talking to systems whose
    errors quote hostnames, credentials, or customer data.
    """
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
    except Exception as e:
        # A handler that raises would otherwise tear down the whole agent
        # loop. Report it as a failed result so the model can react to it
        # the same way it reacts to a validation error. BaseException
        # (KeyboardInterrupt, SystemExit) is deliberately left to propagate.
        logger.exception("Tool handler %r raised", tool_name)
        return ToolResult(
            success=False,
            error=describe_handler_failure(tool_name, e),
        )
    return ToolResult(success=True, content=result)
