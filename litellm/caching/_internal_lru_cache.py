from functools import lru_cache
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def lru_cache_wrapper(
    maxsize: Optional[int] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Wrapper for lru_cache that caches success and exceptions
    """

    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        cached_f = lru_cache(maxsize=maxsize)

        # Use tuple packing to minimize function call overhead
        @cached_f
        def wrapper(*args, **kwargs):
            try:
                return ("success", f(*args, **kwargs))
            except Exception as e:
                return ("error", e)

        # Use local vars for fast-path and direct tuple unpack for performance
        def wrapped(*args, **kwargs):
            result = wrapper(*args, **kwargs)
            status, value = result
            if status == "error":
                raise value
            return value

        return wrapped

    return decorator
