import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

model_name = os.environ.get("OPENAI_MODEL", "gpt-5.6")
client = OpenAI()

response = client.responses.create(
    model=model_name,
    input="Reply with exactly: API connection successful",
)

print(response.output_text)