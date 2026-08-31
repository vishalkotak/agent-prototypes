import json
import os
import logging
from time import perf_counter
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

logger = logging.getLogger(__name__)

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
        logger.info(
            "event=openai_request_started "
            "model=%s continuing=%s",
            self._model_name,
            self._previous_response_id is not None,
        )

        started_at = perf_counter()
        response = self._client.responses.create(**request)
        duration_ms = (perf_counter() - started_at) * 1000
        logger.info(
            "event=openai_request_completed "
            "model=%s response_id=%s duration_ms=%.1f "
            "output_items=%d",
            self._model_name,
            response.id,
            duration_ms,
            len(response.output),
        )

        self._previous_response_id = response.id
        for item in response.output:
            if item.type == "function_call":
                logger.info(
                    "event=openai_function_call "
                    "response_id=%s call_id=%s tool=%s",
                    response.id,
                    item.call_id,
                    item.name,
                )
                return ToolCall(
                    call_id=item.call_id,
                    tool_name=item.name,
                    arguments=json.loads(item.arguments),
                )
        if response.output_text:
            logger.info(
                "event=openai_final_answer response_id=%s",
                response.id,
            )
            return FinalAnswer(content=response.output_text)
        raise RuntimeError(
            "Model returned neither a function call nor a final answer"
        )

        