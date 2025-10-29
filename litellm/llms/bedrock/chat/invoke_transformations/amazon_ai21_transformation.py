import types
from typing import List, Optional

from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)


class AmazonAI21Config(AmazonInvokeConfig, BaseConfig):
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=j2-ultra

    Supported Params for the Amazon / AI21 models:

    - `maxTokens` (int32): The maximum number of tokens to generate per result. Optional, default is 16. If no `stopSequences` are given, generation stops after producing `maxTokens`.

    - `temperature` (float): Modifies the distribution from which tokens are sampled. Optional, default is 0.7. A value of 0 essentially disables sampling and results in greedy decoding.

    - `topP` (float): Used for sampling tokens from the corresponding top percentile of probability mass. Optional, default is 1. For instance, a value of 0.9 considers only tokens comprising the top 90% probability mass.

    - `stopSequences` (array of strings): Stops decoding if any of the input strings is generated. Optional.

    - `frequencyPenalty` (object): Placeholder for frequency penalty object.

    - `presencePenalty` (object): Placeholder for presence penalty object.

    - `countPenalty` (object): Placeholder for count penalty object.
    """

    maxTokens: Optional[int] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    stopSequences: Optional[list] = None
    frequencePenalty: Optional[dict] = None
    presencePenalty: Optional[dict] = None
    countPenalty: Optional[dict] = None

    def __init__(
        self,
        maxTokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topP: Optional[float] = None,
        stopSequences: Optional[list] = None,
        frequencePenalty: Optional[dict] = None,
        presencePenalty: Optional[dict] = None,
        countPenalty: Optional[dict] = None,
    ) -> None:
        # Avoid creating locals().copy(); directly set attributes for provided arguments
        # Only set on the class if not None
        # This preserves behavioral equivalence with the original (note: this still mutates the class, not self)
        if maxTokens is not None:
            setattr(self.__class__, "maxTokens", maxTokens)
        if temperature is not None:
            setattr(self.__class__, "temperature", temperature)
        if topP is not None:
            setattr(self.__class__, "topP", topP)
        if stopSequences is not None:
            setattr(self.__class__, "stopSequences", stopSequences)
        if frequencePenalty is not None:
            setattr(self.__class__, "frequencePenalty", frequencePenalty)
        if presencePenalty is not None:
            setattr(self.__class__, "presencePenalty", presencePenalty)
        if countPenalty is not None:
            setattr(self.__class__, "countPenalty", countPenalty)

        AmazonInvokeConfig.__init__(self)

    @classmethod
    def get_config(cls):
        # Fast-path: avoid repeated attribute lookups and function checks per key, minimize isinstance calls
        blacklist = (
            types.FunctionType,
            types.BuiltinFunctionType,
            classmethod,
            staticmethod,
        )
        d = cls.__dict__
        result = {}
        for k in d:
            if k.startswith("__") or k.startswith("_abc"):
                continue
            v = d[k]
            # Avoid creating a tuple each time; use blacklist and check None first
            if v is None:
                continue
            # Use type(v) instead of isinstance; slightly faster for known function types
            vt = type(v)
            if vt in blacklist:
                continue
            result[k] = v
        return result

    def get_supported_openai_params(self, model: str) -> List:
        return [
            "max_tokens",
            "temperature",
            "top_p",
            "stream",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for k, v in non_default_params.items():
            if k == "max_tokens":
                optional_params["maxTokens"] = v
            if k == "temperature":
                optional_params["temperature"] = v
            if k == "top_p":
                optional_params["topP"] = v
            if k == "stream":
                optional_params["stream"] = v
        return optional_params
