import datetime
from typing import Any, Optional, Union

from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.llm_response_utils.get_api_base import get_api_base
from litellm.litellm_core_utils.logging_utils import LiteLLMLoggingObject
from litellm.types.utils import (
    EmbeddingResponse,
    HiddenParams,
    ModelResponse,
    TranscriptionResponse,
)
from functools import lru_cache


class ResponseMetadata:
    """
    Handles setting and managing `_hidden_params`, `response_time_ms`, and `litellm_overhead_time_ms` for LiteLLM responses
    """

    def __init__(self, result: Any):
        self.result = result
        # If _hidden_params is None, fallback to empty dict
        val = getattr(result, "_hidden_params", {})
        self._hidden_params: Union[HiddenParams, dict] = val if val is not None else {}
        # Cache for _get_value_from_hidden_params
        self._hidden_params_cache = {}

    @property
    def supports_response_time(self) -> bool:
        """Check if response type supports timing metrics"""
        return (
            isinstance(self.result, ModelResponse)
            or isinstance(self.result, EmbeddingResponse)
            or isinstance(self.result, TranscriptionResponse)
        )

    def set_hidden_params(self, logging_obj: LiteLLMLoggingObject, model: Optional[str], kwargs: dict) -> None:
        """Set hidden parameters on the response"""
        # Inline: Avoid repeating .get/.or by extracting once
        model_info = kwargs.get("model_info")
        if not model_info:
            model_info = {}
        model_id = model_info.get("id", None)
        # Use lru_cache for expensive get_api_base, avoid making unhashable objects part of the cache key
        model_for_api_base = model or ""
        # tuple of sorted (key, value) pairs with hashable values; non-hashable values are excluded (safe)
        kwargs_tuple = tuple(
            sorted((k, v) for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, type(None))))
        )
        api_base = self._cached_get_api_base(model_for_api_base, model_id, kwargs_tuple)
        # Avoid repeated property and function calls, use direct assignment
        litellm_call_id = getattr(logging_obj, "litellm_call_id", None)
        # _response_cost_calculator is reasonably fast but can be expensive, so evaluate once
        response_cost = logging_obj._response_cost_calculator(
            result=self.result, litellm_model_name=model, router_model_id=model_id
        )
        # If additional_headers is present in hidden params, process it once
        original_additional_headers = self._get_value_from_hidden_params("additional_headers")
        processed_headers = process_response_headers(original_additional_headers if original_additional_headers else {})
        new_params = {
            "litellm_call_id": litellm_call_id,
            "api_base": api_base,
            "model_id": model_id,
            "response_cost": response_cost,
            "additional_headers": processed_headers,
            "litellm_model_name": model,
        }
        self._update_hidden_params(new_params)

    def _update_hidden_params(self, new_params: dict) -> None:
        """
        Update hidden params - handles when self._hidden_params is a dict or HiddenParams object
        """
        # Handle both dict and HiddenParams cases
        if isinstance(self._hidden_params, dict):
            self._hidden_params.update(new_params)
        elif isinstance(self._hidden_params, HiddenParams):
            # For HiddenParams object, set attributes individually
            for key, value in new_params.items():
                setattr(self._hidden_params, key, value)

    def _get_value_from_hidden_params(self, key: str) -> Optional[Any]:
        """Get value from hidden params - handles when self._hidden_params is a dict or HiddenParams object"""
        if isinstance(self._hidden_params, dict):
            return self._hidden_params.get(key, None)
        elif isinstance(self._hidden_params, HiddenParams):
            return getattr(self._hidden_params, key, None)

    def set_timing_metrics(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        logging_obj: LiteLLMLoggingObject,
    ) -> None:
        """Set response timing metrics"""
        total_response_time_ms = (end_time - start_time).total_seconds() * 1000

        # Set total response time if supported
        if self.supports_response_time:
            self.result._response_ms = total_response_time_ms

        #########################################################
        # 1. Add _response_ms total duration
        #########################################################
        self._update_hidden_params(
            {
                "_response_ms": total_response_time_ms,
            }
        )

        #########################################################
        # 2. Add LiteLLM overhead duration (fast-path if available)
        #########################################################
        llm_api_details = logging_obj.model_call_details
        llm_api_duration_ms = llm_api_details.get("llm_api_duration_ms")
        if llm_api_duration_ms is not None:
            overhead_ms = round(total_response_time_ms - llm_api_duration_ms, 4)
            self._update_hidden_params(
                {
                    "litellm_overhead_time_ms": overhead_ms,
                }
            )

        #########################################################
        # 3. Add duration for reading from cache
        # In this case overhead from litellm is the difference between the cache read duration and the total response time
        #########################################################
        caching_details = logging_obj.caching_details
        # Inline gets to avoid attribute lookup
        if caching_details is not None and caching_details.get("cache_hit") is True:
            cache_duration_ms = caching_details.get("cache_duration_ms")
            if cache_duration_ms is not None:
                overhead_ms = total_response_time_ms - cache_duration_ms
                self._update_hidden_params(
                    {
                        "litellm_overhead_time_ms": overhead_ms,
                    }
                )

    def apply(self) -> None:
        """Apply metadata to the response object"""
        if hasattr(self.result, "_hidden_params"):
            self.result._hidden_params = self._hidden_params

    @staticmethod
    @lru_cache(maxsize=128)
    def _cached_get_api_base(model: str, model_id: Optional[str], kwargs_tuple: tuple) -> Optional[str]:
        # Called from set_hidden_params for performance – kwargs_tuple is a tuple of sorted (k, v) items with hashable values
        # We reconstruct kwargs and pass to get_api_base
        # model_id is just used to make the cache tuple unique per model
        # NOTE: Don't mutate kwargs here!
        kwargs = dict(kwargs_tuple)
        return get_api_base(model=model, optional_params=kwargs)

    @property
    def supports_response_time(self) -> bool:
        # Avoid attribute lookup in timing metrics loop when possible
        # If .result has '_response_ms' attr, it's likely to support this
        return hasattr(self.result, "_response_ms")


def update_response_metadata(
    result: Any,
    logging_obj: LiteLLMLoggingObject,
    model: Optional[str],
    kwargs: dict,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> None:
    """
    Updates response metadata including hidden params and timing metrics
    Updates response metadata, adds the following:
        - response._hidden_params
        - response._hidden_params["litellm_overhead_time_ms"]
        - response.response_time_ms
    """
    if result is None:
        return

    metadata = ResponseMetadata(result)
    metadata.set_hidden_params(logging_obj, model, kwargs)
    metadata.set_timing_metrics(start_time, end_time, logging_obj)
    metadata.apply()
