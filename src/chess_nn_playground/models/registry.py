from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, MutableMapping
from functools import lru_cache
from typing import Any

from torch import nn

from chess_nn_playground.models._registry_manifest import MODEL_SPECS

ModelBuilder = Callable[[dict[str, Any]], nn.Module]

_RUNTIME_BUILDERS: dict[str, ModelBuilder] = {}
_RESOLVED_BUILDERS: dict[str, ModelBuilder] = {}


def _load_manifest_builder(name: str) -> ModelBuilder | None:
    spec = MODEL_SPECS.get(name)
    if spec is None:
        return None
    if name not in _RESOLVED_BUILDERS:
        module_name, attr_name = spec
        module = importlib.import_module(module_name)
        builder = getattr(module, attr_name)
        _RESOLVED_BUILDERS[name] = builder
    return _RESOLVED_BUILDERS[name]


def _make_research_packet_builder(model_name: str) -> ModelBuilder:
    def build_named_research_packet(config: dict[str, Any]) -> nn.Module:
        from chess_nn_playground.models.research_packet_probe import (
            build_research_packet_probe_from_config,
            infer_mechanism_family,
        )

        packet_config = dict(config)
        packet_config.setdefault("name", model_name)
        packet_config.setdefault("packet_profile", model_name)
        packet_config.setdefault("mechanism_family", infer_mechanism_family(model_name))
        return build_research_packet_probe_from_config(packet_config)

    build_named_research_packet.__name__ = f"build_{model_name}_from_config"
    build_named_research_packet.__qualname__ = build_named_research_packet.__name__
    return build_named_research_packet


def _make_bt4_mixer_alias(mixer_name: str) -> ModelBuilder:
    def build_bt4_mixer_alias(config: dict[str, Any]) -> nn.Module:
        from chess_nn_playground.models.architecture.bt4_primitive_mixer import (
            build_bt4_primitive_mixer_from_config,
        )

        alias_config = dict(config)
        alias_config.setdefault("mixer", mixer_name)
        return build_bt4_primitive_mixer_from_config(alias_config)

    build_bt4_mixer_alias.__name__ = f"build_bt4_{mixer_name}_mixer_from_config"
    build_bt4_mixer_alias.__qualname__ = build_bt4_mixer_alias.__name__
    return build_bt4_mixer_alias


@lru_cache(maxsize=1)
def _research_packet_model_names() -> tuple[str, ...]:
    from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES

    return tuple(RESEARCH_PACKET_MODEL_NAMES)


@lru_cache(maxsize=1)
def _bt4_mixer_names() -> tuple[str, ...]:
    try:
        from chess_nn_playground.models.architecture.bt4_mixers import available_mixers

        return tuple(available_mixers())
    except Exception:  # pragma: no cover - mixer package optional / partially built
        return ()


def _bt4_alias_to_mixer(name: str) -> str | None:
    prefix = "bt4_"
    suffix = "_mixer"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    mixer_name = name[len(prefix) : -len(suffix)]
    if mixer_name in _bt4_mixer_names():
        return mixer_name
    return None


def _all_model_names() -> set[str]:
    names = set(MODEL_SPECS)
    names.update(_RUNTIME_BUILDERS)
    names.update(_research_packet_model_names())
    names.update(f"bt4_{mixer_name}_mixer" for mixer_name in _bt4_mixer_names())
    return names


def _resolve_builder(name: str) -> ModelBuilder:
    runtime_builder = _RUNTIME_BUILDERS.get(name)
    if runtime_builder is not None:
        return runtime_builder

    manifest_builder = _load_manifest_builder(name)
    if manifest_builder is not None:
        return manifest_builder

    if name in _research_packet_model_names():
        if name not in _RESOLVED_BUILDERS:
            _RESOLVED_BUILDERS[name] = _make_research_packet_builder(name)
        return _RESOLVED_BUILDERS[name]

    mixer_name = _bt4_alias_to_mixer(name)
    if mixer_name is not None:
        if name not in _RESOLVED_BUILDERS:
            _RESOLVED_BUILDERS[name] = _make_bt4_mixer_alias(mixer_name)
        return _RESOLVED_BUILDERS[name]

    raise KeyError(name)


class _ModelBuilderMapping(MutableMapping[str, ModelBuilder]):
    """Compatibility view over lazily resolved model builders."""

    def __getitem__(self, name: str) -> ModelBuilder:
        try:
            return _resolve_builder(name)
        except KeyError as exc:
            raise KeyError(name) from exc

    def __setitem__(self, name: str, builder: ModelBuilder) -> None:
        _register_runtime_model(name, builder)

    def __delitem__(self, name: str) -> None:
        if name not in _RUNTIME_BUILDERS:
            raise KeyError(name)
        del _RUNTIME_BUILDERS[name]
        _RESOLVED_BUILDERS.pop(name, None)

    def __iter__(self) -> Iterator[str]:
        return iter(available_models())

    def __len__(self) -> int:
        return len(_all_model_names())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in _all_model_names()


MODEL_BUILDERS: MutableMapping[str, ModelBuilder] = _ModelBuilderMapping()


def _register_runtime_model(name: str, builder: ModelBuilder) -> ModelBuilder:
    if not name:
        raise ValueError("Model name must be non-empty")
    if name in _all_model_names() and name not in _RUNTIME_BUILDERS:
        raise ValueError(f"Model already registered: {name}")
    if name in _RUNTIME_BUILDERS:
        raise ValueError(f"Model already registered: {name}")
    _RUNTIME_BUILDERS[name] = builder
    _RESOLVED_BUILDERS[name] = builder
    return builder


def register_model(name: str, builder: ModelBuilder | None = None) -> Any:
    """Register a model builder at runtime.

    Can be used either as ``register_model("name", builder)`` or as a decorator:

        @register_model("name")
        def build_name_from_config(config: dict[str, Any]) -> nn.Module: ...
    """

    if builder is not None:
        return _register_runtime_model(name, builder)

    def _decorator(decorated_builder: ModelBuilder) -> ModelBuilder:
        return _register_runtime_model(name, decorated_builder)

    return _decorator


def available_models() -> list[str]:
    return sorted(_all_model_names())


def build_model(name: str, config: dict[str, Any]) -> nn.Module:
    try:
        builder = _resolve_builder(name)
    except KeyError as exc:
        raise ValueError(f"Unknown model: {name}. Available: {available_models()}") from exc
    return builder(config)
