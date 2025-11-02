from typing import Optional, Tuple, Union

import litellm
from litellm.constants import MIN_NON_ZERO_TEMPERATURE
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str


class DeepInfraConfig(OpenAIGPTConfig):
    """
    Reference: https://deepinfra.com/docs/advanced/openai_api

    The class `DeepInfra` provides configuration for the DeepInfra's Chat Completions API interface. Below are the parameters:
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "deepinfra"

    frequency_penalty: Optional[int] = None
    function_call: Optional[Union[str, dict]] = None
    functions: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        function_call: Optional[Union[str, dict]] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        response_format: Optional[dict] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> None:
        # Optimize locals collection to avoid copy and unneeded computation.
        # Use direct attribute assignment to class for non-None values.
        if frequency_penalty is not None:
            setattr(self.__class__, "frequency_penalty", frequency_penalty)
        if function_call is not None:
            setattr(self.__class__, "function_call", function_call)
        if functions is not None:
            setattr(self.__class__, "functions", functions)
        if logit_bias is not None:
            setattr(self.__class__, "logit_bias", logit_bias)
        if max_tokens is not None:
            setattr(self.__class__, "max_tokens", max_tokens)
        if n is not None:
            setattr(self.__class__, "n", n)
        if presence_penalty is not None:
            setattr(self.__class__, "presence_penalty", presence_penalty)
        if stop is not None:
            setattr(self.__class__, "stop", stop)
        if temperature is not None:
            setattr(self.__class__, "temperature", temperature)
        if top_p is not None:
            setattr(self.__class__, "top_p", top_p)
        if response_format is not None:
            setattr(self.__class__, "response_format", response_format)
        if tools is not None:
            setattr(self.__class__, "tools", tools)
        if tool_choice is not None:
            setattr(self.__class__, "tool_choice", tool_choice)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str):
        supported_openai_params = [
            "stream",
            "frequency_penalty",
            "function_call",
            "functions",
            "logit_bias",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "presence_penalty",
            "stop",
            "temperature",
            "top_p",
            "response_format",
            "tools",
            "tool_choice",
        ]

        if litellm.supports_reasoning(
            model=model,
            custom_llm_provider=self.custom_llm_provider,
        ):
            supported_openai_params.append("reasoning_effort")
        return supported_openai_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        # Optimize lookup by converting supported params to a set (fast containment test)
        supported_openai_params_set = set(self.get_supported_openai_params(model=model))
        # Avoid repeated attribute access / function calls
        drop_params_value = litellm.drop_params is True or drop_params is True

        # Loop in local scope (fastest dict iteration), only handle cases needed
        for param, value in non_default_params.items():
            if (
                param == "temperature" and value == 0 and model == "mistralai/Mistral-7B-Instruct-v0.1"
            ):  # this model does no support temperature == 0
                value = MIN_NON_ZERO_TEMPERATURE  # close to 0
            if param == "tool_choice":
                if value != "auto" and value != "none":
                    # UNSUPPORTED TOOL CHOICE VALUE
                    if drop_params_value:
                        value = None
                    else:
                        raise litellm.utils.UnsupportedParamsError(
                            message="Deepinfra doesn't support tool_choice={}. To drop unsupported openai params from the call, set `litellm.drop_params = True`".format(
                                value
                            ),
                            status_code=400,
                        )
            elif param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params_set:
                if value is not None:
                    optional_params[param] = value
        return optional_params

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # deepinfra is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
        api_base = api_base or get_secret_str("DEEPINFRA_API_BASE") or "https://api.deepinfra.com/v1/openai"
        dynamic_api_key = api_key or get_secret_str("DEEPINFRA_API_KEY")
        return api_base, dynamic_api_key
