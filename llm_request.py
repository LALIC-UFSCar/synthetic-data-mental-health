import json
from typing import Union

from groq import Groq
from openai import OpenAI


def fill_placeholders(prompt: str, values: dict) -> str:
    """Fill in the placeholders in the prompt with the provided values.

    Args:
        prompt (str): Prompt with placeholders
        values (dict): Values to fill in the placeholders

    Returns:
        str: Prompt with filled placeholders
    """
    return prompt.format(**values)


def get_answer(client: Union[Groq, OpenAI], prompt: str, user_input: str,
               config: dict, model: str, response_format: dict = None) -> str:
    """Get the answer from the LLM based on the provided prompt and user input.

    Args:
        client (Union[Groq, OpenAI]): Client for the LLM
        prompt (str): System prompt for the LLM
        user_input (str): User input for the LLM
        config (dict): Configuration settings for the LLM
        model (str): Model name for the LLM
        response_format (dict, optional): Response format settings for the LLM. Defaults to None.

    Returns:
        str: The answer from the LLM
    """
    messages = [{'role': 'system', 'content': prompt},
                {'role': 'user', 'content': user_input}]

    kwargs = {
        'messages': messages,
        'model': model,
        'temperature': config['temperature'],
        'top_p': config['top_p'],
        'stop': None,
        'max_completion_tokens': config['max_completion_tokens'],
        'stream': False
    }

    if response_format:
        kwargs['response_format'] = response_format

    if client.__class__.__module__.startswith("openai"):
        # OpenAI uses 'max_tokens'
        kwargs.pop("max_completion_tokens", None)

    answer = client.chat.completions.create(**kwargs)

    if response_format:
        return json.loads(answer.choices[0].message.content)

    return answer.choices[0].message.content.strip()
