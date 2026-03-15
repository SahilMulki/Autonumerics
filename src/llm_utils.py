# llm_utils.py

from openai import OpenAI

# Initialize OpenAI client (expects OPENAI_API_KEY in env)
client = OpenAI()


def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4.1") -> str:
    """
    Simple wrapper around OpenAI chat API.
    Returns the assistant's text content as a string.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content
