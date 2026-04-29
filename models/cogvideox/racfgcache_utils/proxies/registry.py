from .base import ProxyBase

_PROXY_REGISTRY = {}


def register_proxy(cls):
    _PROXY_REGISTRY[cls.name] = cls
    return cls


def get_proxy(name: str) -> ProxyBase:
    if name not in _PROXY_REGISTRY:
        raise KeyError(f"Unknown proxy: {name}. Available: {list(_PROXY_REGISTRY.keys())}")
    return _PROXY_REGISTRY[name]()