"""Lazy actor factory for native VERL and vendored AReaL rollouts."""

from collections.abc import Mapping
from typing import Any


def _get_child(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(value, key, default)


def config_get(config: Any, dotted_path: str, default: Any = None) -> Any:
    """Read DictConfig, mapping, or namespace values without resolving imports."""

    value = config
    sentinel = object()
    for key in dotted_path.split("."):
        value = _get_child(value, key, sentinel)
        if value is sentinel:
            return default
    return value


def resolve_rollout_backend_name(config_or_name: Any) -> str:
    """Return the canonical backend name and reject silent fallbacks."""

    if isinstance(config_or_name, str):
        name = config_or_name
    else:
        name = config_get(config_or_name, "async_training.rollout_backend", "verl")
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "verl": "verl",
        "native": "verl",
        "areal": "areal",
        "a_real": "areal",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted({"verl", "areal"}))
        raise ValueError(f"Unknown fully-async rollout backend {name!r}; expected one of: {supported}") from exc


def get_rollout_backend_actor_class(config_or_name: Any):
    """Resolve a Ray actor class while keeping AReaL imports backend-local."""

    backend = resolve_rollout_backend_name(config_or_name)
    if backend == "verl":
        from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter

        return FullyAsyncRollouter

    from .areal_rollouter import AReaLFullyAsyncRollouter

    return AReaLFullyAsyncRollouter


def create_rollout_backend(config, tokenizer, processor=None, device_name=None):
    """Create the configured rollout Ray actor with the shared constructor API."""

    actor_class = get_rollout_backend_actor_class(config)
    return actor_class.remote(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        device_name=device_name,
    )
