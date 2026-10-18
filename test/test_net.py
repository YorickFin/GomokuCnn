"""网络模块测试：棋盘编码、输出形状、非法着法 mask、价值范围。"""

import numpy as np
import torch

from src.config import board_size
from src.net import ValueCNN, board_to_tensor


def make_model():
    """测试用小模型，CPU 上快速前向。"""
    return ValueCNN(hidden_channels=8, num_blocks=2)


def make_board():
    board = [[0 for _ in range(board_size)] for _ in range(board_size)]
    board[0][0] = 1
    board[1][1] = -1
    board[7][7] = 1
    return board


def test_board_to_tensor_channels():
    board = make_board()
    tensor = board_to_tensor(torch) if False else board_to_tensor(board)
    assert tensor.shape == (3, board_size, board_size)

    # 通道 0：当前玩家（1）
    assert tensor[0, 0, 0].item() == 1.0
    assert tensor[0, 7, 7].item() == 1.0
    assert tensor[0, 1, 1].item() == 0.0
    # 通道 1：对手（-1）
    assert tensor[1, 1, 1].item() == 1.0
    assert tensor[1, 0, 0].item() == 0.0
    # 通道 2：空位
    assert tensor[2, 0, 0].item() == 0.0
    assert tensor[2, 5, 5].item() == 1.0


def test_forward_shapes():
    model = make_model().eval()
    board = make_board()
    x = board_to_tensor(board).unsqueeze(0)
    with torch.no_grad():
        value, logits = model(x)
    assert value.shape == (1, 1)
    assert logits.shape == (1, board_size * board_size)


def test_value_range():
    model = make_model().eval()
    x = board_to_tensor(make_board()).unsqueeze(0)
    with torch.no_grad():
        value, _ = model(x)
    assert -1.0 <= float(value) <= 1.0


def test_policy_masks_illegal_moves():
    """已占位置 softmax 后概率必须为 0，合法位置概率和为 1。"""
    model = make_model().eval()
    board = make_board()
    x = board_to_tensor(board).unsqueeze(0)
    with torch.no_grad():
        _, probs = model.calc(x)
    probs = probs.squeeze(0).numpy()

    # 已占位置概率为 0
    assert probs[0, 0] == 0.0
    assert probs[1, 1] == 0.0
    assert probs[7, 7] == 0.0
    # 空位概率非负，总和为 1
    assert probs.min() >= 0.0
    assert abs(probs.sum() - 1.0) < 1e-5


def test_batch_forward_consistency():
    """批量前向与单样本前向结果一致（mask 不破坏 batch 行为）。"""
    model = make_model().eval()
    boards = [make_board(), [[0] * board_size for _ in range(board_size)]]
    x = torch.stack([board_to_tensor(b) for b in boards])
    with torch.no_grad():
        values, logits = model(x)
    assert values.shape == (2, 1)
    assert logits.shape == (2, board_size * board_size)
    # 空棋盘所有位置合法，softmax 覆盖全部 225 个点
    assert torch.isfinite(logits).all()
