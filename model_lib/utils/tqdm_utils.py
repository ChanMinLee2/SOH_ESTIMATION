"""tqdm wrapper: provides a no-op fallback when tqdm is not installed."""

from __future__ import annotations
from typing import Any, Iterable

try:
    from tqdm import tqdm as _tqdm

    def tqdm(iterable: Iterable = None, **kwargs: Any) -> Any:  # type: ignore[misc]
        return _tqdm(iterable, **kwargs)

    def trange(*args: Any, **kwargs: Any) -> Any:
        from tqdm import trange as _trange
        return _trange(*args, **kwargs)

    def write(msg: str) -> None:
        _tqdm.write(msg)

except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable = None, **kwargs: Any) -> Any:  # type: ignore[misc]
        return iterable if iterable is not None else iter([])

    def trange(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        return range(*args)

    def write(msg: str) -> None:  # type: ignore[misc]
        print(msg)
