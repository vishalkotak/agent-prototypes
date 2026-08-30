from triage_agent.openai_model import OpenAIModel
from triage_agent.tools import TOOL_REGISTRY


model = OpenAIModel(TOOL_REGISTRY)

decision = model.decide(
    [
        {
            "role": "user",
            "content": "Investigate elevated checkout errors.",
        }
    ]
)

print(decision)