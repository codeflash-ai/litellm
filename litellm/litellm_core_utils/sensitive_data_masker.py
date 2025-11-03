from typing import Any, Dict, Optional, Set

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH_SENSITIVE_DATA_MASKER


class SensitiveDataMasker:
    def __init__(
        self,
        sensitive_patterns: Optional[Set[str]] = None,
        visible_prefix: int = 4,
        visible_suffix: int = 4,
        mask_char: str = "*",
    ):
        self.sensitive_patterns = sensitive_patterns or {
            "password",
            "secret",
            "key",
            "token",
            "auth",
            "credential",
            "access",
            "private",
            "certificate",
            "fingerprint",
            "tenancy",
        }

        self.visible_prefix = visible_prefix
        self.visible_suffix = visible_suffix
        self.mask_char = mask_char

        # Precompute sensitive_patterns as a set for faster lookup, but preserve code style
        self._sensitive_patterns_set = set(self.sensitive_patterns)

    def _mask_value(self, value: str) -> str:
        value_str = str(value)
        val_len = len(value_str)
        prefix = self.visible_prefix
        suffix = self.visible_suffix

        # Avoid unnecessary str conversion, check value once
        if not value or val_len < (prefix + suffix):
            return value

        masked_length = val_len - (prefix + suffix)

        if suffix == 0:
            # Avoid multiple concatenations by using join for mask
            return f"{value_str[:prefix]}{self.mask_char * masked_length}"
        else:
            return f"{value_str[:prefix]}{self.mask_char * masked_length}{value_str[-suffix:]}"

    def is_sensitive_key(self, key: str) -> bool:
        # Fast path: avoid str conversion if key is already str and lower
        if type(key) is str:
            key_lower = key.lower()
        else:
            key_lower = str(key).lower()
        # Split on underscores and check if any segment matches the pattern
        # This avoids false positives like "max_tokens" matching "token"
        # but still catches "api_key", "access_token", etc.
        key_segments = key_lower.replace("-", "_").split("_")

        # Use set intersection rather than 'in' in loop
        # This drastically reduces how many pattern comparisons are made
        return bool(self._sensitive_patterns_set.intersection(key_segments))

    def mask_dict(
        self,
        data: Dict[str, Any],
        depth: int = 0,
        max_depth: int = DEFAULT_MAX_RECURSE_DEPTH_SENSITIVE_DATA_MASKER,
    ) -> Dict[str, Any]:
        if depth >= max_depth:
            return data

        masked_data: Dict[str, Any] = {}
        # Minor: localise functions/attrs for speedup
        is_sensitive_key = self.is_sensitive_key
        _mask_value = self._mask_value
        # Avoid repeated isinstance allocation
        basic_types = (int, float, bool, str, list)

        for k, v in data.items():
            try:
                if isinstance(v, dict):
                    masked_data[k] = self.mask_dict(v, depth + 1)
                elif hasattr(v, "__dict__") and not isinstance(v, type):
                    masked_data[k] = self.mask_dict(vars(v), depth + 1)
                elif is_sensitive_key(k):
                    # Fast path: skip str() conversion if v is already str
                    str_value = v if isinstance(v, str) else str(v) if v is not None else ""
                    masked_data[k] = _mask_value(str_value)
                else:
                    # Avoid tuple allocation per iteration: use single isinstance
                    if isinstance(v, basic_types):
                        masked_data[k] = v
                    else:
                        masked_data[k] = str(v)
            except Exception:
                masked_data[k] = "<unable to serialize>"

        return masked_data


# Usage example:
"""
masker = SensitiveDataMasker()
data = {
    "api_key": "sk-1234567890abcdef",
    "redis_password": "very_secret_pass",
    "port": 6379,
    "tags": ["East US 2", "production", "test"]
}
masked = masker.mask_dict(data)
# Result: {
#    "api_key": "sk-1****cdef",
#    "redis_password": "very****pass",
#    "port": 6379,
#    "tags": ["East US 2", "production", "test"]
# }
"""
