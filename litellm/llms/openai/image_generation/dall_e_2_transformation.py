from typing import List

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams


class DallE2ImageGenerationConfig(BaseImageGenerationConfig):
    """
    OpenAI dall-e-2 image generation config
    """

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return ["n", "response_format", "quality", "size", "user"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        supported_params_set = set(supported_params)
        optional_keys = set(optional_params.keys())

        for k in non_default_params:
            if k not in optional_keys:
                if k in supported_params_set:
                    optional_params[k] = non_default_params[k]
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                    )

        return optional_params
