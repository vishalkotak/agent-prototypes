# agent-prototypes

Prototypes for LLM agent systems — tool dispatch, agent loops, and the plumbing
between a model and the things it is allowed to do.

## Prototypes

| Directory | What it explores |
| --- | --- |
| [`triage-agent/`](triage-agent/) | An incident-triage agent: a schema-validated tool registry, a model-agnostic agent loop, and a binding to the OpenAI Responses API. |

## Layout

Each prototype is a self-contained Python project in its own directory, with its
own `pyproject.toml` and test suite. They share nothing but this repository, so
one prototype's dependencies never constrain another's.

## Conventions

- **Environment.** A single `.venv/` at the repository root, with each prototype
  installed into it as needed (`pip install -e <prototype>/`).
- **Secrets.** Real credentials live in a git-ignored `.env` at the repository
  root; each prototype commits a `.env.example` documenting the keys it reads.
- **Ignores.** One `.gitignore` at the root covers every prototype.
