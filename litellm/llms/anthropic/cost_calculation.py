"""
Helper util for handling anthropic-specific cost calculation
- e.g.: prompt caching
"""

from typing import TYPE_CHECKING, Optional, Tuple

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import SearchContextCostPerQuery

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage


def cost_per_token(model: str, usage: "Usage") -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    return generic_cost_per_token(model=model, usage=usage, custom_llm_provider="anthropic")


def get_cost_for_anthropic_web_search(
    model_info: Optional["ModelInfo"] = None,
    usage: Optional["Usage"] = None,
) -> float:
    """
    Get the cost of using a web search tool for Anthropic.
    """
    if (
        model_info is None
        or usage is None
        or usage.server_tool_use is None
        or usage.server_tool_use.web_search_requests is None
    ):
        return 0.0

    # Get the cost per web search request
    search_context_pricing: SearchContextCostPerQuery = (
        model_info.get("search_context_cost_per_query") or SearchContextCostPerQuery()
    )
    cost_per_web_search_request = search_context_pricing.get("search_context_size_medium", 0.0)
    if not cost_per_web_search_request:
        return 0.0

    # Calculate the total cost
    return cost_per_web_search_request * usage.server_tool_use.web_search_requests
