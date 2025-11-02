"""
Transformation logic from OpenAI /v1/embeddings format to LM Studio's  `/v1/embeddings` format. 

Why separate file? Make it easy to see how transformation works

Docs - https://lmstudio.ai/docs/basics/server
"""

import types
from typing import List


class LmStudioEmbeddingConfig:
    """
    Reference: https://lmstudio.ai/docs/basics/server
    """

    def __init__(
        self,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        # Optimize: Avoid tuple allocation inside loop, discard unnecessary isinstance tests once per value.
        # - Minor perf: move filter funcs out of dictcomp
        ignore_types = (types.FunctionType, types.BuiltinFunctionType, classmethod, staticmethod)
        cls_dict = cls.__dict__

        # Pre-filter keys and values to avoid checking every key starting with '__'
        # Move not None outside the isinstance cascade for slightly faster short-circuiting
        # Use direct dict comprehension for best performance, as list conversion has unnecessary cost
        return {
            k: v
            for k, v in cls_dict.items()
            if (not k.startswith("__") and v is not None and not isinstance(v, ignore_types))
        }

    def get_supported_openai_params(self) -> List[str]:
        return []

    def map_openai_params(self, non_default_params: dict, optional_params: dict) -> dict:
        return optional_params
