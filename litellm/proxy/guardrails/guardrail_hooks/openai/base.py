from typing import TYPE_CHECKING, List, Optional
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)

if TYPE_CHECKING:
    from litellm.types.llms.openai import AllMessageValues


class OpenAIGuardrailBase:
    """
    Base class for OpenAI guardrails.
    """

    def get_user_prompt(self, messages: List["AllMessageValues"]) -> Optional[str]:
        """
        Get the last consecutive block of messages from the user.

        Example:
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm good, thank you!"},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ]
        get_user_prompt(messages) -> "What is the weather in Tokyo?"
        """

        if not messages:
            return None

        # Find the last consecutive block of user messages from the end
        end = len(messages)
        start = end
        for i in range(end - 1, -1, -1):
            if messages[i].get("role") == "user":
                start = i
            else:
                break

        if start == end:
            return None

        user_messages = messages[start:end]
        # Accumulate prompt lines efficiently
        lines = [convert_content_list_to_str(msg) for msg in user_messages]
        result = "\n".join(lines).strip()
        return result if result else None
