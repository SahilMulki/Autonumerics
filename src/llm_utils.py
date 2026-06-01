# llm_utils.py

import anthropic

# Initialize Anthropic client (expects ANTHROPIC_API_KEY in env)
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
