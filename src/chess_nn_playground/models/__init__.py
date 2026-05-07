"""Registered chess puzzle model definitions."""

from chess_nn_playground.models.cnn import SimpleChessCNN
from chess_nn_playground.models.blocker_pin_lattice import BlockerPinLatticeNetwork
from chess_nn_playground.models.empty_square_opportunity_network import EmptySquareOpportunityNetwork
from chess_nn_playground.models.global_scratchpad_boardnet import GlobalScratchpadBoardNet
from chess_nn_playground.models.hypercolumn_square_readout_cnn import HypercolumnSquareReadoutCNN
from chess_nn_playground.models.independence_residual import IndependenceResidualInteractionNetwork
from chess_nn_playground.models.latent_reply_entropy import LatentReplyEntropyNetwork
from chess_nn_playground.models.lc0_bt4 import LC0BT4Classifier
from chess_nn_playground.models.mlp import BoardMLP
from chess_nn_playground.models.multiplicative_conjunction_convnet import MultiplicativeConjunctionConvNet
from chess_nn_playground.models.nnue import StockfishStyleNNUE
from chess_nn_playground.models.residual_calibration import ResidualCalibrationErrorField
from chess_nn_playground.models.residual_cnn import ResidualChessCNN
from chess_nn_playground.models.safe_reply_certificate import SafeReplyCertificateVerifier
from chess_nn_playground.models.set_query_attention import SetQueryAttentionBottleneck

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
