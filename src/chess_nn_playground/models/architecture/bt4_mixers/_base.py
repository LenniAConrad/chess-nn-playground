"""Mixer registry for bt4_primitive_mixer.

A *mixer* is the swappable spatial-mixing operator inside a BT4-style residual
block. The repo's `lc0_bt4` block mixes spatially with a pair of 3x3 convs;
this package lets that mixing operator be swapped for a chess-aware primitive.

Contract for every mixer module:

    forward(x: Tensor[B, C, 8, 8]) -> Tensor[B, C, 8, 8]

i.e. channels and the 8x8 board shape are preserved. The BT4 block wraps the
mixer with SqueezeExcite + residual + activation, so the mixer only has to do
the spatial interaction.

Each mixer file in this package registers a builder:

    @register_mixer("my_mixer")
    def build(channels: int, **kwargs) -> nn.Module: ...

`build_mixer(name, channels, **kwargs)` looks one up. `available_mixers()`
lists them. Sibling modules are auto-imported on first use so a new
`bt4_mixers/<name>.py` file is discovered without editing this file.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable

from torch import nn

MixerBuilder = Callable[..., nn.Module]

_MIXER_BUILDERS: dict[str, MixerBuilder] = {}
_DISCOVERED = False


def register_mixer(name: str) -> Callable[[MixerBuilder], MixerBuilder]:
    def _decorator(builder: MixerBuilder) -> MixerBuilder:
        if name in _MIXER_BUILDERS:
            raise ValueError(f"mixer {name!r} already registered")
        _MIXER_BUILDERS[name] = builder
        return builder

    return _decorator


def _discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    package = importlib.import_module(__name__.rsplit(".", 1)[0])
    for mod in pkgutil.iter_modules(package.__path__):
        if mod.name.startswith("_"):
            continue
        importlib.import_module(f"{package.__name__}.{mod.name}")


def available_mixers() -> list[str]:
    _discover()
    return sorted(_MIXER_BUILDERS)


def build_mixer(name: str, channels: int, **kwargs: Any) -> nn.Module:
    _discover()
    if name not in _MIXER_BUILDERS:
        raise ValueError(
            f"mixer {name!r} is not registered. Available: {sorted(_MIXER_BUILDERS)}"
        )
    return _MIXER_BUILDERS[name](channels=channels, **kwargs)
