from typing import List, Optional, Union

from httpx import Headers

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ..base_llm.chat.transformation import BaseLLMException


class FireworksAIException(BaseLLMException):
    pass


class FireworksAIMixin:
    """
    Common Base Config functions across Fireworks AI Endpoints
    """

    def get_error_class(self, error_message: str, status_code: int, headers: Union[dict, Headers]) -> BaseLLMException:
        return FireworksAIException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def _get_api_key(self, api_key: Optional[str]) -> Optional[str]:
        if api_key:
            return api_key
        for key in (
            "FIREWORKS_API_KEY",
            "FIREWORKS_AI_API_KEY",
            "FIREWORKSAI_API_KEY",
            "FIREWORKS_AI_TOKEN",
        ):
            secret = get_secret_str(key)
            if secret:
                return secret
        return None

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        api_key = self._get_api_key(api_key)
        if api_key is None:
            raise ValueError("FIREWORKS_API_KEY is not set")

        return {"Authorization": f"Bearer {api_key}", **headers}
