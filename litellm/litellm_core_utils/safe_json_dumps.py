import json
from typing import Any

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH


def safe_dumps(data: Any, max_depth: int = DEFAULT_MAX_RECURSE_DEPTH) -> str:
    """
    Recursively serialize data while detecting circular references.
    If a circular reference is detected then a marker string is returned.
    """

    # Prebinding for perf
    _str = str
    _isinstance = isinstance
    _id = id

    def _serialize(obj: Any, seen: set, depth: int) -> Any:
        # Check for maximum depth.
        if depth > max_depth:
            return "MaxDepthExceeded"
        # Base-case: if it is a primitive, simply return it.
        if _isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        ident = _id(obj)
        # Check for circular reference.
        if ident in seen:
            return "CircularReference Detected"
        seen.add(ident)
        try:
            if _isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    # Only allow str keys (no attempt at conversion for speed & safety)
                    if _isinstance(k, str):
                        result[k] = _serialize(v, seen, depth + 1)
                return result
            elif _isinstance(obj, list):
                return [_serialize(item, seen, depth + 1) for item in obj]
            elif _isinstance(obj, tuple):
                return tuple(_serialize(item, seen, depth + 1) for item in obj)
            elif _isinstance(obj, set):
                return sorted([_serialize(item, seen, depth + 1) for item in obj])
            else:
                # Fall back to string conversion for non-serializable objects.
                try:
                    return _str(obj)
                except Exception:
                    return "Unserializable Object"
        finally:
            seen.remove(ident)

    safe_data = _serialize(data, set(), 0)
    return json.dumps(safe_data, default=str)
