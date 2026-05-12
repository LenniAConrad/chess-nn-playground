from __future__ import annotations

import pytest
import torch

from chess_nn_playground.models.trunk.idea_blocks import BoardConvStem, BoardTensorSpec, GlobalPoolClassifier, require_board_tensor


def test_idea_blocks_preserve_expected_classifier_shape():
    stem = BoardConvStem(input_channels=18, channels=8, depth=2)
    head = GlobalPoolClassifier(input_channels=stem.output_channels, num_classes=1)

    logits = head(stem(torch.zeros(4, 18, 8, 8)))

    assert logits.shape == (4, 1)


def test_require_board_tensor_rejects_wrong_shape():
    with pytest.raises(ValueError, match="Expected board tensor"):
        require_board_tensor(torch.zeros(4, 17, 8, 8), BoardTensorSpec(input_channels=18))
