def _convert_prompt_to_chat_messages(prompt: str, metadata: dict = None) -> list[dict[str, str]]:
    """
    Convert a plain text prompt into a chat message format.

    The prompt is wrapped as a single user message.

    Args:
        prompt (str): The input prompt text.
        metadata (dict): A dictionary containing the metadata of the prompt.
    Returns:
        list[dict[str, str]]: A list containing a single message dict.
    """
    return {"messages": [{"role": "user", "content": prompt}], "metadata": metadata}


def convert_prompts_to_chat_messages(prompts: list[str]) -> list[list[dict[str, str]]]:
    """
    Convert a list of plain text prompts into chat message format.

    Each prompt is wrapped as a single user message.

    Args:
        prompts (list[str]): A list of input prompt texts.
    Returns:
        list[list[dict[str, str]]]: A list of message lists.
    """
    return [_convert_prompt_to_chat_messages(p) for p in prompts]
