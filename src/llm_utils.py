# llm_utils.py

import os
import pathlib
import anthropic


def _load_dot_env() -> None:
    """Load KEY=VALUE pairs from .env in the project root into os.environ.

    Uses no third-party dependencies — reads the file directly. Only sets keys
    that are not already present in the environment (os.environ takes priority),
    matching the behaviour of python-dotenv's override=False mode.
    """
    env_path = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as fh:
        for raw in fh:
            line = raw.strip()
            # Skip blank lines and comments
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip()
            # Strip optional surrounding quotes (single or double)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            # Only inject if not already set — shell export takes priority
            if key and key not in os.environ:
                os.environ[key] = value


# Load before constructing the client so ANTHROPIC_API_KEY is available
_load_dot_env()

# Initialize Anthropic client (reads ANTHROPIC_API_KEY from os.environ)
client = anthropic.Anthropic()


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 8096,
) -> str:
    """
    Simple wrapper around Anthropic messages API.
    Returns the assistant's text content as a string.
    """
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.content[0].text
