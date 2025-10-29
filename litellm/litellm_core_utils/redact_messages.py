# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import copy
from typing import TYPE_CHECKING, Any, Optional

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.secret_managers.main import str_to_bool
from litellm.types.utils import StandardCallbackDynamicParams
import asyncio

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as _LiteLLMLoggingObject,
    )

    LiteLLMLoggingObject = _LiteLLMLoggingObject
else:
    LiteLLMLoggingObject = Any


def redact_message_input_output_from_custom_logger(
    litellm_logging_obj: LiteLLMLoggingObject, result, custom_logger: CustomLogger
):
    if (
        hasattr(custom_logger, "message_logging")
        and custom_logger.message_logging is not True
    ):
        return perform_redaction(litellm_logging_obj.model_call_details, result)
    return result


def perform_redaction(model_call_details: dict, result):
    """
    Performs the actual redaction on the logging object and result.
    """
    # Redact model_call_details
    model_call_details["messages"] = [
        {"role": "user", "content": "redacted-by-litellm"}
    ]
    model_call_details["prompt"] = ""
    model_call_details["input"] = ""

    stream_flag = model_call_details.get("stream", False)
    complete_streaming_response = model_call_details.get(
        "complete_streaming_response", None
    )
    if stream_flag is True and complete_streaming_response is not None:
        _streaming_response = complete_streaming_response
        choices = getattr(_streaming_response, "choices", None)
        if choices is not None:
            # In-place redaction, avoid hasattr + isinstance chain where possible
            for choice in choices:
                # Fast type check by direct comparison where possible
                # litellm.Choices and litellm.utils.StreamingChoices are likely classes
                if type(choice) is litellm.Choices:
                    choice.message.content = "redacted-by-litellm"
                elif type(choice) is litellm.utils.StreamingChoices:
                    choice.delta.content = "redacted-by-litellm"
                else:
                    # Fallback to isinstance if subclassing is possible (preserves original logic)
                    if isinstance(choice, litellm.Choices):
                        choice.message.content = "redacted-by-litellm"
                    elif isinstance(choice, litellm.utils.StreamingChoices):
                        choice.delta.content = "redacted-by-litellm"
        else:
            output = getattr(_streaming_response, "output", None)
            if output is not None:
                for output_item in output:
                    content = getattr(output_item, "content", None)
                    if isinstance(content, list):
                        for content_part in content:
                            text_attr = getattr(content_part, "text", None)
                            if text_attr is not None:
                                content_part.text = "redacted-by-litellm"

    # Redact result
    if result is not None:
        # Check if result is a coroutine, async generator, or other async object - these cannot be deepcopied
        # Try to minimize attribute lookups and function calls
        is_async = (
            asyncio.iscoroutine(result)
            or asyncio.iscoroutinefunction(result)
            or hasattr(result, "__aiter__")
            or hasattr(result, "__anext__")  # async generator  # async iterator
        )
        if is_async:
            return {"text": "redacted-by-litellm"}

        # For non-async results, avoid unnecessary deepcopy for simple types
        # Check if result is one of known types; for unknown types fallback to deepcopy + dictionary response (original logic)
        # However, preserve the original behavioral order and coverage

        # Only deepcopy if we expect fields to redact
        result_type = type(result)
        if result_type in (
            litellm.ModelResponse,
            litellm.ResponsesAPIResponse,
            litellm.EmbeddingResponse,
        ) or isinstance(
            result,
            (
                litellm.ModelResponse,
                litellm.ResponsesAPIResponse,
                litellm.EmbeddingResponse,
            ),
        ):
            _result = copy.deepcopy(result)
            if isinstance(_result, litellm.ModelResponse):
                choices = getattr(_result, "choices", None)
                if choices:
                    for choice in choices:
                        if type(choice) is litellm.Choices:
                            choice.message.content = "redacted-by-litellm"
                        elif type(choice) is litellm.utils.StreamingChoices:
                            choice.delta.content = "redacted-by-litellm"
                        else:
                            if isinstance(choice, litellm.Choices):
                                choice.message.content = "redacted-by-litellm"
                            elif isinstance(choice, litellm.utils.StreamingChoices):
                                choice.delta.content = "redacted-by-litellm"
            elif isinstance(_result, litellm.ResponsesAPIResponse):
                output = getattr(_result, "output", None)
                if output:
                    for output_item in output:
                        content = getattr(output_item, "content", None)
                        if isinstance(content, list):
                            for content_part in content:
                                text_attr = getattr(content_part, "text", None)
                                if text_attr is not None:
                                    content_part.text = "redacted-by-litellm"
            elif isinstance(_result, litellm.EmbeddingResponse):
                data = getattr(_result, "data", None)
                if data is not None:
                    _result.data = []
            else:
                return {"text": "redacted-by-litellm"}
            return _result
        else:
            return {"text": "redacted-by-litellm"}


def should_redact_message_logging(model_call_details: dict) -> bool:
    """
    Determine if message logging should be redacted.
    """
    _request_headers = (
        model_call_details.get("litellm_params", {}).get("metadata", {}) or {}
    )

    request_headers = _request_headers.get("headers", {})

    possible_request_headers = [
        "litellm-enable-message-redaction",  # old header. maintain backwards compatibility
        "x-litellm-enable-message-redaction",  # new header
    ]

    is_redaction_enabled_via_header = False
    for header in possible_request_headers:
        if bool(request_headers.get(header, False)):
            is_redaction_enabled_via_header = True
            break

    # check if user opted out of logging message/response to callbacks
    if (
        litellm.turn_off_message_logging is not True
        and is_redaction_enabled_via_header is not True
        and _get_turn_off_message_logging_from_dynamic_params(model_call_details)
        is not True
    ):
        return False

    if request_headers and bool(
        request_headers.get("litellm-disable-message-redaction", False)
    ):
        return False

    # user has OPTED OUT of message redaction
    if _get_turn_off_message_logging_from_dynamic_params(model_call_details) is False:
        return False

    return True


def redact_message_input_output_from_logging(
    model_call_details: dict, result, input: Optional[Any] = None
) -> Any:
    """
    Removes messages, prompts, input, response from logging. This modifies the data in-place
    only redacts when litellm.turn_off_message_logging == True
    """
    if should_redact_message_logging(model_call_details):
        return perform_redaction(model_call_details, result)
    return result


def _get_turn_off_message_logging_from_dynamic_params(
    model_call_details: dict,
) -> Optional[bool]:
    """
    gets the value of `turn_off_message_logging` from the dynamic params, if it exists.

    handles boolean and string values of `turn_off_message_logging`
    """
    standard_callback_dynamic_params: Optional[
        StandardCallbackDynamicParams
    ] = model_call_details.get("standard_callback_dynamic_params", None)
    if standard_callback_dynamic_params:
        _turn_off_message_logging = standard_callback_dynamic_params.get(
            "turn_off_message_logging"
        )
        if isinstance(_turn_off_message_logging, bool):
            return _turn_off_message_logging
        elif isinstance(_turn_off_message_logging, str):
            return str_to_bool(_turn_off_message_logging)
    return None


def redact_user_api_key_info(metadata: dict) -> dict:
    """
    removes any user_api_key_info before passing to logging object, if flag set

    Usage:

    SDK
    ```python
    litellm.redact_user_api_key_info = True
    ```

    PROXY:
    ```yaml
    litellm_settings:
        redact_user_api_key_info: true
    ```
    """
    if litellm.redact_user_api_key_info is not True:
        return metadata

    new_metadata = {}
    for k, v in metadata.items():
        if isinstance(k, str) and k.startswith("user_api_key"):
            pass
        else:
            new_metadata[k] = v

    return new_metadata
