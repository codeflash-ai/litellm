from typing import List, Union, cast

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    AllPromptValues,
    OpenAITextCompletionUserMessage,
)


def is_tokens_or_list_of_tokens(value: List):
    # Check if it's a list of integers (tokens) or a list of lists of integers (list of tokens)
    if not isinstance(value, list):
        return False
    if not value:
        return False
    first_item = value[0]
    if isinstance(first_item, int):
        # Assume homogeneous list, check all ints
        return all(isinstance(item, int) for item in value)
    elif isinstance(first_item, list):
        # Assume homogeneous list of lists of ints
        return all(
            isinstance(item, list) and all(isinstance(i, int) for i in item)
            for item in value
        )
    return False


def _transform_prompt(
    messages: Union[List[AllMessageValues], List[OpenAITextCompletionUserMessage]],
) -> AllPromptValues:
    if len(messages) == 1:  # base case
        message_content = messages[0].get("content")
        if (
            message_content
            and isinstance(message_content, list)
            and is_tokens_or_list_of_tokens(message_content)
        ):
            openai_prompt: AllPromptValues = cast(AllPromptValues, message_content)
        else:
            content = convert_content_list_to_str(cast(AllMessageValues, messages[0]))
            openai_prompt = content
    else:
        # Use list comprehension for efficiency
        prompt_str_list: List[str] = [
            convert_content_list_to_str(cast(AllMessageValues, m)) for m in messages
        ]
        openai_prompt = prompt_str_list
    return openai_prompt
