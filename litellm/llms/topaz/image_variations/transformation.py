import base64
import time
from io import BytesIO
from typing import Any, List, Mapping, Optional, Tuple, Union

from aiohttp import ClientResponse
from httpx import Headers, Response

from litellm.llms.base_llm.chat.transformation import (
    BaseLLMException,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import OpenAIImageVariationOptionalParams
from litellm.types.utils import (
    FileTypes,
    HttpHandlerRequestFields,
    ImageObject,
    ImageResponse,
)

from ...base_llm.image_variations.transformation import BaseImageVariationConfig
from ..common_utils import TopazException, TopazModelInfo


class TopazImageVariationConfig(TopazModelInfo, BaseImageVariationConfig):
    def get_supported_openai_params(self, model: str) -> List[OpenAIImageVariationOptionalParams]:
        return ["response_format", "size"]

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = api_base or "https://api.topazlabs.com"
        return f"{api_base}/image/v1/enhance"

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for k, v in non_default_params.items():
            if k == "response_format":
                optional_params["output_format"] = v
            elif k == "size":
                split_v = v.split("x")
                assert len(split_v) == 2, "size must be in the format of widthxheight"
                optional_params["output_width"] = split_v[0]
                optional_params["output_height"] = split_v[1]
        return optional_params

    def prepare_file_tuple(
        self,
        file_data: FileTypes,
    ) -> Tuple[str, Optional[FileTypes], str, Mapping[str, str]]:
        """
        Convert various file input formats to a consistent tuple format for HTTPX
        Returns: (filename, file_content, content_type, headers)
        """
        # Default values
        filename = "image.png"
        content_type = "image/png"
        # Reuse empty dict instead of allocating each time since output never mutates it
        _empty_headers: Mapping[str, str] = {}

        # Short-circuit for most common & fastest cases: bytes or BytesIO

        if isinstance(file_data, (bytes, BytesIO)):
            return (filename, file_data, content_type, _empty_headers)

        if isinstance(file_data, tuple):
            l = len(file_data)
            # Fast variable assignment using constants for index lookup
            if l == 2:
                fn, content = file_data
                # The 'or' fallback is needed per original behavior
                return ((fn or filename), content, content_type, _empty_headers)
            elif l == 3:
                fn, content, ct = file_data
                return ((fn or filename), content, (ct or content_type), _empty_headers)
            elif l == 4:
                fn, content, ct, headers = file_data
                return ((fn or filename), content, (ct or content_type), headers)

        # Fallback to defaults
        return (filename, None, content_type, _empty_headers)

    def transform_request_image_variation(
        self,
        model: Optional[str],
        image: FileTypes,
        optional_params: dict,
        headers: dict,
    ) -> HttpHandlerRequestFields:
        request_params = HttpHandlerRequestFields(
            files={"image": self.prepare_file_tuple(image)},
            data=optional_params,
        )

        return request_params

    def _common_transform_response_image_variation(
        self,
        image_content: bytes,
        response_ms: float,
    ) -> ImageResponse:
        # Convert to base64
        base64_image = base64.b64encode(image_content).decode("utf-8")

        return ImageResponse(
            created=int(time.time()),
            data=[
                ImageObject(
                    b64_json=base64_image,
                    url=None,
                    revised_prompt=None,
                )
            ],
            response_ms=response_ms,
        )

    async def async_transform_response_image_variation(
        self,
        model: Optional[str],
        raw_response: ClientResponse,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        image: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
    ) -> ImageResponse:
        image_content = await raw_response.read()

        response_ms = logging_obj.get_response_ms()

        return self._common_transform_response_image_variation(image_content, response_ms)

    def transform_response_image_variation(
        self,
        model: Optional[str],
        raw_response: Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        image: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
    ) -> ImageResponse:
        image_content = raw_response.content

        response_ms = raw_response.elapsed.total_seconds() * 1000  # Convert to milliseconds

        return self._common_transform_response_image_variation(image_content, response_ms)

    def get_error_class(self, error_message: str, status_code: int, headers: Union[dict, Headers]) -> BaseLLMException:
        return TopazException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
