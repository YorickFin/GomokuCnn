"""自对弈数据管线测试：温度退火、D4 增强、单局标签、按局切分、多进程生成。"""

import random

import numpy as np
import torch

from src.board import evaluation_func
from src.config import Config, board_size
from src.net import ValueCNN
from src.utils import (WeightedDataset, apply_d4, augment_data, concat_games,
                       generate_fixed_eval_set, generate_random_board,
                       generate_selfplay_data, generate_single_game,
                       load_fixed_eval_set, save_fixed_eval_set, split_games)


def empty_board_list():
    return [[0 for _ in range(board_size)] for _ in range(board_size)]


def make_tiny_model():
    return ValueCNN(hidden_channels=8, num_blocks=2).eval()


# ---------- 随机开局视角 ----------

def test_random_board_perspective_and_valid():
    """回归测试：随机开局后 1 必须为当前行动方。

    1 行动时子数关系为 count(-1) - count(1) ∈ {0, 1}
    （偶数手后双方相等，奇数手翻转后对方多一子）。
    """
    model = make_tiny_model()
    for _ in range(5):
        board = generate_random_board(model)
        c1 = sum(row.count(1) for row in board)
        c2 = sum(row.count(-1) for row in board)
        assert c2 - c1 in (0, 1), f"视角错误: count(1)={c1}, count(-1)={c2}"
        # 开局不应直接是终局
        assert evaluation_func(board) == 0


def test_random_board_empty_prob(monkeypatch):
    """开局强化：empty_prob=1 必返回空盘；empty_prob=0 返回合法视角棋盘。"""
    model = make_tiny_model()

    monkeypatch.setattr(Config, "opening_empty_prob", 1.0)
    board = generate_random_board(model)
    assert all(v == 0 for row in board for v in row), "empty_prob=1 应返回空盘"

    monkeypatch.setattr(Config, "opening_empty_prob", 0.0)
    for _ in range(3):
        board = generate_random_board(model)
        c1 = sum(row.count(1) for row in board)
        c2 = sum(row.count(-1) for row in board)
        assert c2 - c1 in (0, 1), f"视角错误: count(1)={c1}, count(-1)={c2}"
        assert evaluation_func(board) == 0


# ---------- 每局随机温度 ----------

def test_per_game_temperature_range():
    """自对弈采用 per-game 随机温度，验证 Config 范围合理。"""
    assert Config.temp_min < Config.temp_max
    assert 0.0 < Config.temp_min <= 0.5   # 下限近贪心
    assert Config.temp_max >= 0.6          # 上限允许充分探索


# ---------- 数据增强 ----------

def _fake_game(n=3):
    """构造 n 个假样本：棋盘与策略对齐（策略只在空位非零）。"""
    boards = torch.zeros(n, 3, board_size, board_size)
    policies = torch.zeros(n, board_size, board_size)
    for k in range(n):
        boards[k, 2] = 1.0  # 全空
        boards[k, 0, 0, k] = 1.0
        boards[k, 2, 0, k] = 0.0
        policies[k] = 1.0 / (board_size * board_size - 1)
        policies[k, 0, k] = 0.0
    values = torch.tensor([1.0, -1.0, 0.0][:n])
    weights = torch.ones(n)
    return boards, policies, values, weights


def test_augment_eightfold():
    boards, policies, values, weights = _fake_game(3)
    ab, ap, av, aw = augment_data(boards, policies, values, weights)
    assert ab.shape[0] == 3 * 8
    assert ap.shape[0] == 3 * 8
    assert av.shape[0] == 3 * 8
    assert aw.shape[0] == 3 * 8


def test_augment_policy_aligned_with_board():
    """增强后 policy 非零位置必须与棋盘空位严格对应，且概率和为 1。"""
    boards, policies, values, weights = _fake_game(2)
    ab, ap, _av, _aw = augment_data(boards, policies, values, weights)
    empty_mask = ab[:, 2] > 0.5
    policy_nonzero = ap > 0
    assert torch.equal(empty_mask, policy_nonzero)
    sums = ap.view(ap.size(0), -1).sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_augment_preserves_value():
    boards, policies, values, weights = _fake_game(2)
    _ab, _ap, av, _aw = augment_data(boards, policies, values, weights)
    # 每个原始样本的 8 个增强副本 value 相同
    for k in range(2):
        assert torch.allclose(av[k * 8:(k + 1) * 8],
                              torch.full((8,), float(values[k])))


# ---------- 在线随机 D4 增强（WeightedDataset） ----------

def test_dataset_no_augment_returns_raw():
    """augment=False（默认）：原样返回，不随机变换。"""
    boards, policies, values, weights = _fake_game(2)
    ds = WeightedDataset(boards, policies, values, weights)
    assert len(ds) == 2
    b, p, v, w = ds[0]
    assert torch.equal(b, boards[0])
    assert torch.equal(p, policies[0])
    assert float(v) == float(values[0])
    assert float(w) == float(weights[0])


def test_dataset_augment_length_not_materialized():
    """augment=True：样本数仍是原始数量（不物化 8 倍）。"""
    boards, policies, values, weights = _fake_game(3)
    ds = WeightedDataset(boards, policies, values, weights, augment=True)
    assert len(ds) == 3


def test_dataset_augment_produces_varied_views():
    """同一样本多次取样应看到不同 D4 视图，且覆盖全部 8 种对称。"""
    boards, policies, values, weights = _fake_game(1)
    ds = WeightedDataset(boards, policies, values, weights, augment=True)

    random.seed(1234)
    views = {tuple(ds[0][0].reshape(-1).tolist()) for _ in range(60)}
    assert len(views) >= 4, f"在线增强视图多样性不足，仅 {len(views)} 种"

    # 枚举 apply_d4 全部 8 种参数，与 augment_data 的 8 个副本集合一致
    ab, ap, _av, _aw = augment_data(boards, policies, values, weights)
    expected_boards = {tuple(ab[i].reshape(-1).tolist()) for i in range(8)}
    online_boards = set()
    for k in range(4):
        for flip in (False, True):
            ob, _op = apply_d4(boards[0], policies[0], k, flip)
            online_boards.add(tuple(ob.reshape(-1).tolist()))
    assert online_boards == expected_boards


def test_dataset_augment_aligned_and_untouched_source():
    """在线增强后 board/policy 对齐、概率和为 1，且原始数据不被就地修改。"""
    boards, policies, values, weights = _fake_game(2)
    boards_before = boards.clone()
    ds = WeightedDataset(boards, policies, values, weights, augment=True)

    random.seed(0)
    for _ in range(20):
        b, p, v, w = ds[1]
        empty_mask = b[2] > 0.5
        assert torch.equal(empty_mask, p > 0)
        assert abs(float(p.sum()) - 1.0) < 1e-5
        # value/weight 不随变换改变
        assert float(v) == float(values[1])
        assert float(w) == float(weights[1])

    # __getitem__ 的 rot90/flip 不在源张量上就地修改
    assert torch.equal(boards, boards_before)


# ---------- 单局自对弈 ----------

def test_generate_single_game_labels(monkeypatch):
    """旧管线单局自对弈：样本长度一致、value 为终局胜负交替、policy 合法。

    显式 collect_all_nodes=False：本测试验证旧标签语义（value 交替回溯、
    weight 全 1）保持不变；全节点收集行为见 test_full_node_collection.py。
    """
    # 固定从空盘开局，跳过随机开局搜索以加速
    monkeypatch.setattr("src.utils.generate_random_board",
                        lambda model, eval_batch_size=None: empty_board_list())

    model = make_tiny_model()
    boards, policies, values, weights = generate_single_game(
        0, model.state_dict(), num_simulations=12,
        hidden_channels=8, num_blocks=2, collect_all_nodes=False)

    n = boards.shape[0]
    assert n >= 5
    assert policies.shape[0] == n
    assert len(values) == n
    assert len(weights) == n

    # 最后一步为最后落子方视角：胜(+1) 或 平(0)，不可能为 -1
    assert values[-1] in (1.0, 0.0)

    # value 沿落子序列交替
    for k in range(n):
        expected = values[-1] * ((-1) ** (n - 1 - k))
        assert abs(values[k] - expected) < 1e-9

    # policy 非零位置必须是空位（已占位概率恒为 0），且每个样本概率和为 1
    empty_mask = boards[:, 2] > 0.5
    assert not (policies[~empty_mask] > 0).any()
    assert (policies >= 0).all()
    sums = policies.view(n, -1).sum(dim=1)
    assert torch.allclose(sums, torch.ones(n), atol=1e-5)

    # 棋盘编码合法：每格恰好在一个通道为 1
    channel_sum = boards.sum(dim=1)
    assert torch.allclose(channel_sum, torch.ones_like(channel_sum))


# ---------- 按局切分 ----------

def test_split_games_no_overlap():
    games = [f"game_{i}" for i in range(10)]
    train, val = split_games(games, train_ratio=0.8)
    assert len(train) + len(val) == 10
    assert set(train).isdisjoint(set(val))


def test_concat_games():
    g1 = _fake_game(3)
    g2 = _fake_game(2)
    boards, policies, values, weights = concat_games([
        (g1[0], g1[1], g1[2].tolist(), g1[3].tolist()),
        (g2[0], g2[1], g2[2].tolist(), g2[3].tolist()),
    ])
    assert boards.shape[0] == 5
    assert policies.shape[0] == 5
    assert values.shape[0] == 5
    assert weights.shape[0] == 5


# ---------- 多进程自对弈（端到端，较慢） ----------

def test_generate_selfplay_data_multiprocess(monkeypatch):
    monkeypatch.setattr(Config, "num_workers", 2)
    monkeypatch.setattr(Config, "train_ratio", 0.5)
    model = make_tiny_model()

    train_data, val_data = generate_selfplay_data(
        model, num_games=2, num_simulations=10)

    for data in (train_data, val_data):
        boards, policies, values, weights = data
        assert boards.shape[0] > 0
        assert boards.shape[1:] == (3, board_size, board_size)
        assert policies.shape[1:] == (board_size, board_size)
        assert boards.shape[0] == policies.shape[0] == values.shape[0]
        # 增强后样本数为 8 的倍数
        assert boards.shape[0] % 8 == 0
        # policy 非零位置必须是空位，且概率总和为 1
        empty_mask = boards[:, 2] > 0.5
        assert not (policies[~empty_mask] > 0).any()
        sums = policies.view(policies.shape[0], -1).sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---------- 固定评估集（跨轮次可比的“考卷”）----------

def test_fixed_eval_set_generate_and_roundtrip(monkeypatch, tmp_path):
    """固定评估集：全部棋局用于评估、增强后落盘可完整读回。"""
    monkeypatch.setattr(Config, "num_workers", 2)
    model = make_tiny_model()

    data = generate_fixed_eval_set(model, num_games=2, num_simulations=10)
    boards, policies, values, weights = data
    assert boards.shape[0] > 0
    assert boards.shape[0] % 8 == 0
    assert boards.shape[0] == policies.shape[0] == values.shape[0]

    path = tmp_path / "sub" / "eval_fixed.pt"
    save_fixed_eval_set(str(path), data)
    assert path.exists()

    loaded = load_fixed_eval_set(str(path))
    assert loaded is not None
    assert loaded[0].shape == boards.shape
    assert torch.allclose(loaded[0], boards)
    assert torch.allclose(loaded[2], values)

    # 缺失文件返回 None，而不是抛异常
    assert load_fixed_eval_set(str(tmp_path / "missing.pt")) is None
