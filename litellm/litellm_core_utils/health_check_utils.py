"""
Utils used for litellm.ahealth_check()
"""


def _filter_model_params(model_params: dict) -> dict:
    """Remove 'messages' param from model params."""
    return {k: v for k, v in model_params.items() if k != "messages"}


def _create_health_check_response(response_headers: dict) -> dict:
    response = {}

    # Store repeated lookups in local variables to minimize dict key search
    remaining_requests = response_headers.get("x-ratelimit-remaining-requests")
    if remaining_requests is not None:
        response["x-ratelimit-remaining-requests"] = remaining_requests

    remaining_tokens = response_headers.get("x-ratelimit-remaining-tokens")
    if remaining_tokens is not None:
        response["x-ratelimit-remaining-tokens"] = remaining_tokens

    ms_region = response_headers.get("x-ms-region")
    if ms_region is not None:
        response["x-ms-region"] = ms_region

    return response
