from __future__ import annotations

from torch import nn

from chess_nn_playground.models import registry
from chess_nn_playground.models.registry import MODEL_BUILDERS, available_models, build_model, register_model


def test_available_models_does_not_resolve_manifest_builders() -> None:
    registry._RESOLVED_BUILDERS.clear()

    models = available_models()

    assert "simple_cnn" in models
    assert "bt4_conv_mixer" in models
    assert "simple_cnn" not in registry._RESOLVED_BUILDERS


def test_build_model_resolves_manifest_builder_lazily() -> None:
    registry._RESOLVED_BUILDERS.clear()

    model = build_model(
        "simple_cnn",
        {
            "input_channels": 18,
            "num_classes": 2,
            "channels": 4,
            "num_blocks": 1,
        },
    )

    assert isinstance(model, nn.Module)
    assert MODEL_BUILDERS["simple_cnn"].__name__ == "build_cnn_from_config"
    assert "simple_cnn" in registry._RESOLVED_BUILDERS


def test_register_model_supports_decorator_registration() -> None:
    name = "__pytest_runtime_registered_model__"

    @register_model(name)
    def build_runtime_model(config: dict) -> nn.Module:
        return nn.Identity()

    try:
        assert name in available_models()
        assert isinstance(build_model(name, {}), nn.Identity)
        assert MODEL_BUILDERS[name] is build_runtime_model
    finally:
        del MODEL_BUILDERS[name]
