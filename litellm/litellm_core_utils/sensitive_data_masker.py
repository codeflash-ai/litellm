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

    def _mask_value(self, value: str) -> str:
        # Avoid double str() call and handle empty string faster
        if not value:
            return value

        value_str = str(value)
        val_len = len(value_str)
        prefix = self.visible_prefix
        suffix = self.visible_suffix

        # Fast-path if too short to mask
        if val_len < (prefix + suffix):
            return value

        masked_length = val_len - (prefix + suffix)
        # Avoid string concatenation overhead with f-strings for large masked_length
        if suffix == 0:
            # Only prefix and mask (no suffix)
            return value_str[:prefix] + (self.mask_char * masked_length)
        else:
            return value_str[:prefix] + (self.mask_char * masked_length) + value_str[-suffix:]

    def is_sensitive_key(self, key: str) -> bool:
        key_lower = str(key).lower()
        # Split on underscores and check if any segment matches the pattern
        # This avoids false positives like "max_tokens" matching "token"
        # but still catches "api_key", "access_token", etc.
        key_segments = key_lower.replace("-", "_").split("_")
        result = any(pattern in key_segments for pattern in self.sensitive_patterns)
        return result

    def mask_dict(
        self,
        data: Dict[str, Any],
        depth: int = 0,
        max_depth: int = DEFAULT_MAX_RECURSE_DEPTH_SENSITIVE_DATA_MASKER,
    ) -> Dict[str, Any]:
        if depth >= max_depth:
            return data

        masked_data: Dict[str, Any] = {}
        for k, v in data.items():
            try:
                if isinstance(v, dict):
                    masked_data[k] = self.mask_dict(v, depth + 1)
                elif hasattr(v, "__dict__") and not isinstance(v, type):
                    masked_data[k] = self.mask_dict(vars(v), depth + 1)
                elif self.is_sensitive_key(k):
                    str_value = str(v) if v is not None else ""
                    masked_data[k] = self._mask_value(str_value)
                else:
                    masked_data[k] = v if isinstance(v, (int, float, bool, str, list)) else str(v)
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
