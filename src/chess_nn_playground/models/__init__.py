"""Registered chess puzzle model definitions."""

from chess_nn_playground.models.cnn import SimpleChessCNN
from chess_nn_playground.models.lc0_bt4 import LC0BT4Classifier
from chess_nn_playground.models.mlp import BoardMLP
from chess_nn_playground.models.nnue import StockfishStyleNNUE
from chess_nn_playground.models.residual_cnn import ResidualChessCNN

__all__ = [
    "BoardMLP",
    "LC0BT4Classifier",
    "ResidualChessCNN",
    "SimpleChessCNN",
    "StockfishStyleNNUE",
]
