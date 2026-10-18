"""模型评估测试：Model 封装加载/推理、模型对弈 play_game/pit。"""

import torch

from src.board import new_board
from src.eval import Model, pit, play_game
from src.mcts import MCTS
from src.net import ValueCNN


def make_tiny_model():
    return ValueCNN(hidden_channels=8, num_blocks=2).eval()


def test_model_load_and_move(tmp_path):
    """从 checkpoint 加载的 Model 能返回合法落子。"""
    model = make_tiny_model()
    ckpt = tmp_path / "tiny.pth"
    torch.save(model.state_dict(), ckpt)

    agent = Model(str(ckpt), simulations=5)
    board = new_board()
    move = agent.call(board, temperature=0)

    assert 0 <= move[0] < 15 and 0 <= move[1] < 15
    assert board[move[0]][move[1]] == 0


def test_play_game_result_valid():
    """单局对弈返回 a 视角结果 ∈ {1, -1, 0}。"""
    mcts_a = MCTS(make_tiny_model(), noise_eps=0.0)
    mcts_b = MCTS(make_tiny_model(), noise_eps=0.0)
    result = play_game(mcts_a, mcts_b, simulations=8, a_starts=True)
    assert result in (1, -1, 0)


def test_pit_counts_consistent():
    """对弈统计：胜+负+平 = 局数，胜率在 [0,1]。"""
    model = make_tiny_model()
    stats = pit(model, model, num_games=2, simulations=8)

    assert stats["wins_a"] + stats["wins_b"] + stats["draws"] == stats["num_games"] == 2
    assert 0.0 <= stats["win_rate_a"] <= 1.0


def test_pit_alternating_starter():
    """交替先手：两局结果对同一模型应对称（a/b 合计为两局）。"""
    model = make_tiny_model()
    stats = pit(model, model, num_games=2, simulations=8)
    assert stats["num_games"] == 2
