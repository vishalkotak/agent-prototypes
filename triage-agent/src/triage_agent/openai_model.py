import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from triage_agent.runtime import (
    FinalAnswer,
    ModelDecision,
    ToolCall,
    ToolCallResult,
)
from triage_agent.tools import ToolDefinition

SYSTEM_INSTRUCTIONS = """
You are an incident-triage agent.

Investigate incidents using the provided read-only tools.
Base conclusions only on evidence returned by tools.
Do not invent metric values or claim that a tool was executed when it was not.
"""

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

class OpenAIModel:
    def __init__(
        self,
        tool_registry: Mapping[str, ToolDefinition],
    ) -> None:
        load_dotenv()
        self._client = OpenAI()
        self._model_name = os.environ.get(
            "OPENAI_MODEL",
            "gpt-5.6",
        )
        self._tools = [
            build_openai_tool(tool)
            for tool in tool_registry.values()
        ]
        self._previous_response_id: str | None = None

    def decide(self, history: list[Any]) -> ModelDecision:
        request: dict[str, Any] = {
            "model": self._model_name,
            "instructions": SYSTEM_INSTRUCTIONS,
            "tools": self._tools,
            "parallel_tool_calls": False,
        }
        if self._previous_response_id is None:
            # First model call: provide the original incident.
            request["input"] = history[0]["content"]
            request["tool_choice"] = "required"
        else:
            # Subsequent call: provide the latest tool result.
            latest_event = history[-1]
            if not isinstance(latest_event, ToolCallResult):
                raise ValueError(
                    "Expected the latest history event to be a ToolCallResult"
                )
            request["previous_response_id"] = self._previous_response_id
            request["tool_choice"] = "auto"
            request["input"] = [
                {
                    "type": "function_call_output",
                    "call_id": latest_event.call_id,
                    "output": json.dumps(
                        asdict(latest_event.result)
                    ),
                }
            ]
        response = self._client.responses.create(**request)
        self._previous_response_id = response.id
        for item in response.output:
            if item.type == "function_call":
                return ToolCall(
                    call_id=item.call_id,
                    tool_name=item.name,
                    arguments=json.loads(item.arguments),
                )
        if response.output_text:
            return FinalAnswer(content=response.output_text)
        raise RuntimeError(
            "Model returned neither a function call nor a final answer"
        )

        