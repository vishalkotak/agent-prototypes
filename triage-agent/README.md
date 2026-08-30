# triage-agent

A tool-dispatch layer and agent loop for an incident-triage agent. The agent is
given read-only tools over service metrics and asked to investigate an incident,
grounding its conclusions in what the tools actually returned.

## Design

Three layers, each independently testable:

| Module | Responsibility |
| --- | --- |
| `tools.py` | The tool registry. Each tool pairs a Pydantic arguments model with a handler, and `execute_tool` validates arguments before dispatch. |
| `runtime.py` | The agent loop. Drives any `ModelClient` through alternating tool calls and results until a `FinalAnswer`, bounded by `RunLimits.max_model_steps`. |
| `openai_model.py` | An OpenAI Responses API implementation of `ModelClient`, chaining turns via `previous_response_id`. |

Two properties are worth calling out:

- **Arguments are validated, not trusted.** Tool argument models inherit from
  `ToolArgs`, which sets `extra="forbid"`. That rejects hallucinated arguments at
  runtime, and makes `model_json_schema()` emit `additionalProperties: false` —
  which is exactly what OpenAI's `strict` function-calling mode requires.
- **The loop does not know which model it is driving.** `ModelDecision` is a
  `ToolCall | FinalAnswer` union, so `ScriptedModel` substitutes for
  `OpenAIModel` and the tests run with no network calls.

## Tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `query_metrics` | `service`, `metric` (one of `latency_ms`, `error_rate`, `request_rate`, `cpu_percent`), `window_minutes` (1–60) | Samples within the window, plus the metric's unit. Backed by fixed data, so runs are reproducible. |

## Setup

From the repository root:

```sh
python3 -m venv .venv                     # Python 3.11+
.venv/bin/pip install -e "triage-agent[dev]"
cp triage-agent/.env.example .env         # then fill in OPENAI_API_KEY
```

`.env` is read from the repository root; `OPENAI_MODEL` selects the model and
defaults to `gpt-5.6`.

## Running

```sh
python scripts/check_openai.py      # verify API connectivity
python scripts/check_tool_call.py   # one model turn, print the decision
python scripts/run_openai_agent.py  # the full agent loop
```

## Tests

```sh
cd triage-agent && pytest
```

The suite covers metric-window filtering and the loop's central contract: that a
tool result is fed back to the model on the following turn, under the same
`call_id`.
