from triage_agent.openai_model import OpenAIModel
from triage_agent.runtime import RunLimits, run_agent
from triage_agent.tools import TOOL_REGISTRY
from triage_agent.logging_config import configure_logging

configure_logging()
model = OpenAIModel(TOOL_REGISTRY)

answer = run_agent(
    model=model,
    user_task="Investigate elevated checkout errors.",
    limits=RunLimits(max_model_steps=8),
)

print("\nFinal answer:\n")
print(answer.content)