"""Name -> factory registry for AFCAlgorithm variants, so adding a new
algorithm later (beyond PEMAFC F/A and PCA) is a one-line registration
instead of touching simulate.py's dispatch logic."""

from typing import Callable

from .base import AFCAlgorithm

_REGISTRY: dict[str, Callable[..., AFCAlgorithm]] = {}


def register_algorithm(name: str):
    def deco(factory: Callable[..., AFCAlgorithm]):
        if name in _REGISTRY:
            raise ValueError(f"Algorithm '{name}' already registered")
        _REGISTRY[name] = factory
        return factory
    return deco


def create_algorithm(name: str, **kwargs) -> AFCAlgorithm:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown algorithm '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def registered_names() -> list[str]:
    return sorted(_REGISTRY)
