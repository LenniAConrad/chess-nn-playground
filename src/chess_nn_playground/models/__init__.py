"""Registered chess puzzle model definitions."""

from chess_nn_playground.models.trunk.cnn import SimpleChessCNN
from chess_nn_playground.models.trunk.blocker_pin_lattice import BlockerPinLatticeNetwork
from chess_nn_playground.models.trunk.empty_square_opportunity_network import EmptySquareOpportunityNetwork
from chess_nn_playground.models.trunk.global_scratchpad_boardnet import GlobalScratchpadBoardNet
from chess_nn_playground.models.trunk.hypercolumn_square_readout_cnn import HypercolumnSquareReadoutCNN
from chess_nn_playground.models.trunk.independence_residual import IndependenceResidualInteractionNetwork
from chess_nn_playground.models.trunk.latent_reply_entropy import LatentReplyEntropyNetwork
from chess_nn_playground.models.trunk.lc0_bt4 import LC0BT4Classifier
from chess_nn_playground.models.trunk.mlp import BoardMLP
from chess_nn_playground.models.trunk.multiplicative_conjunction_convnet import MultiplicativeConjunctionConvNet
from chess_nn_playground.models.trunk.nnue import StockfishStyleNNUE
from chess_nn_playground.models.trunk.residual_calibration import ResidualCalibrationErrorField
from chess_nn_playground.models.trunk.residual_cnn import ResidualChessCNN
from chess_nn_playground.models.trunk.safe_reply_certificate import SafeReplyCertificateVerifier
from chess_nn_playground.models.trunk.set_query_attention import SetQueryAttentionBottleneck

__all__ = [
    "BoardMLP",
    "BlockerPinLatticeNetwork",
    "EmptySquareOpportunityNetwork",
    "GlobalScratchpadBoardNet",
    "HypercolumnSquareReadoutCNN",
    "IndependenceResidualInteractionNetwork",
    "LatentReplyEntropyNetwork",
    "LC0BT4Classifier",
    "MultiplicativeConjunctionConvNet",
    "ResidualCalibrationErrorField",
    "ResidualChessCNN",
    "SafeReplyCertificateVerifier",
    "SetQueryAttentionBottleneck",
    "SimpleChessCNN",
    "StockfishStyleNNUE",
]
