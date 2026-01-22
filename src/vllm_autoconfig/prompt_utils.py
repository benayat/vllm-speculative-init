from copy import deepcopy

def _convert_prompt_to_chat_messages(prompt: str, metadata: dict = None) -> dict[str, dict]:
    """
    Convert a plain text prompt into a chat message format.

    The prompt is wrapped as a single user message.

    Args:
        prompt (str): The input prompt text.
        metadata (dict): A dictionary containing the metadata of the prompt.
    Returns:
        dict[str, dict]: A dict with 'messages' list and 'metadata' dict.
    """
    # Create a deep copy of metadata to avoid mutations
    metadata_copy = deepcopy(metadata) if metadata is not None else {}

    return {"messages": [{"role": "user", "content": prompt}], "metadata": metadata_copy}


def convert_prompts_to_chat_messages(prompts: list[str]) -> list[dict[str, dict]]:
    """
    Convert a list of plain text prompts into chat message format.

    Each prompt is wrapped as a single user message.
    Creates deep copies to avoid mutating original data.

    Args:
        prompts (list[str]): A list of input prompt texts.
    Returns:
        list[dict[str, dict]]: A list of message dicts, each with 'messages' and 'metadata'.
    """
    # Create a deep copy of the prompts list to avoid any mutation issues
    prompts_copy = deepcopy(prompts)
    return [_convert_prompt_to_chat_messages(p) for p in prompts_copy]
