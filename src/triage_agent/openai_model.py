from typing import Any

from triage_agent.tools import ToolDefinition

def build_openai_tool(
    tool: ToolDefinition,
) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.arguments_model.model_json_schema(),
        "strict": True,
    }