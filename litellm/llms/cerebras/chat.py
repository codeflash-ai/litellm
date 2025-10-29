"""
Cerebras Chat Completions API

this is OpenAI compatible - no translation needed / occurs
"""

from typing import Optional

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class CerebrasConfig(OpenAIGPTConfig):
    """
    Reference: https://inference-docs.cerebras.ai/api-reference/chat-completions

    Below are the parameters:
    """

    max_tokens: Optional[int] = None
    response_format: Optional[dict] = None
    seed: Optional[int] = None
    stream: Optional[bool] = None
    top_p: Optional[int] = None
    tool_choice: Optional[str] = None
    tools: Optional[list] = None
    user: Optional[str] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        stop: Optional[str] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[int] = None,
        tool_choice: Optional[str] = None,
        tools: Optional[list] = None,
        user: Optional[str] = None,
    ) -> None:
        # Assign attributes directly rather than mutating the class or copying all locals
        # This avoids unnecessary memory allocation and side effects
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if response_format is not None:
            self.response_format = response_format
        if seed is not None:
            self.seed = seed
        if stop is not None:
            self.stop = stop
        if stream is not None:
            self.stream = stream
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if tool_choice is not None:
            self.tool_choice = tool_choice
        if tools is not None:
            self.tools = tools
        if user is not None:
            self.user = user

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for the given model

        """
        # Returning static list directly instead of constructing on every call
        return [
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "seed",
            "stop",
            "stream",
            "temperature",
            "top_p",
            "tool_choice",
            "tools",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model)
        # Convert to set for O(1) lookup instead of repeated list search
        supported_openai_params_set = set(supported_openai_params)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params_set:
                optional_params[param] = value
        return optional_params
