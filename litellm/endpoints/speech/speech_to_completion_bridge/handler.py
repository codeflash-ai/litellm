"""
Handler for transforming /chat/completions api requests to litellm.responses requests
"""

from typing import TYPE_CHECKING, Optional, Union

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from litellm import LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent


class SpeechToCompletionBridgeHandlerInputKwargs(TypedDict):
    model: str
    input: str
    voice: Optional[Union[str, dict]]
    optional_params: dict
    litellm_params: dict
    logging_obj: "LiteLLMLoggingObj"
    headers: dict
    custom_llm_provider: str


class SpeechToCompletionBridgeHandler:
    def __init__(self):
        from .transformation import SpeechToCompletionBridgeTransformationHandler

        super().__init__()
        self.transformation_handler = SpeechToCompletionBridgeTransformationHandler()

    def validate_input_kwargs(self, kwargs: dict) -> SpeechToCompletionBridgeHandlerInputKwargs:
        # Pull all required keys at once
        required_str_keys = ("model", "custom_llm_provider", "input")
        required_dict_keys = ("optional_params", "litellm_params", "headers")
        # Assign variables to local scope in one pass to reduce lookups
        missing_str = [k for k in required_str_keys if not (isinstance(kwargs.get(k), str))]
        if missing_str:
            raise ValueError(f"{missing_str[0]} is required")
        missing_dict = [k for k in required_dict_keys if not (isinstance(kwargs.get(k), dict))]
        if missing_dict:
            raise ValueError(f"{missing_dict[0]} is required")

        # Only call get once per key
        model = kwargs["model"]
        custom_llm_provider = kwargs["custom_llm_provider"]
        input = kwargs["input"]
        optional_params = kwargs["optional_params"]
        litellm_params = kwargs["litellm_params"]
        headers = kwargs["headers"]

        # No duplicate header checking
        from litellm import LiteLLMLoggingObj

        logging_obj = kwargs.get("logging_obj")
        if logging_obj is None or not isinstance(logging_obj, LiteLLMLoggingObj):
            raise ValueError("logging_obj is required")

        return SpeechToCompletionBridgeHandlerInputKwargs(
            model=model,
            input=input,
            voice=kwargs.get("voice"),
            optional_params=optional_params,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
            headers=headers,
        )

    def speech(
        self,
        model: str,
        input: str,
        voice: Optional[Union[str, dict]],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str,
    ) -> "HttpxBinaryResponseContent":
        received_args = locals()
        from litellm import completion
        from litellm.types.utils import ModelResponse

        validated_kwargs = self.validate_input_kwargs(received_args)
        model = validated_kwargs["model"]
        input = validated_kwargs["input"]
        optional_params = validated_kwargs["optional_params"]
        litellm_params = validated_kwargs["litellm_params"]
        headers = validated_kwargs["headers"]
        logging_obj = validated_kwargs["logging_obj"]
        custom_llm_provider = validated_kwargs["custom_llm_provider"]
        voice = validated_kwargs["voice"]

        request_data = self.transformation_handler.transform_request(
            model=model,
            input=input,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
            litellm_logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
            voice=voice,
        )

        result = completion(
            **request_data,
        )

        if isinstance(result, ModelResponse):
            return self.transformation_handler.transform_response(
                model_response=result,
            )
        else:
            raise Exception("Unmapped response type. Got type: {}".format(type(result)))


speech_to_completion_bridge_handler = SpeechToCompletionBridgeHandler()
