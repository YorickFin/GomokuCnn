"""infer_batch_size 参数扫描：速度 vs 搜索质量（生产 cpp 后端）。

权衡：批量越大 GPU 吞吐越高，但批量评估返回前的“投机模拟”看不到
更早批次的搜索结果（信息延迟），batch 接近模拟数时搜索退化为
近似随机扩展。因此对每个候选批次同时测量：
1. 速度：单局自对弈每手平均耗时（真实网络、训练模拟数）；
2. 质量：均势残局（均匀先验 mock）下致胜点是否被贪心选中及访问占比，
   绝对标准 = 200 模拟必须命中致胜点。

属于性能基准，运行约 1~2 分钟：
    .venv\\Scripts\\python.exe -m pytest test/test_benchmark_batch_sweep.py -s
"""

import copy
import logging
import time

import pytest

from src.board import new_board
from src.config import Config, board_size
from src.mcts import MCTS, calc_next_move
from src.net import ValueCNN, get_device
from src.utils import generate_single_game
import src.mcts as mcts_mod

logger = logging.getLogger(__name__)

CANDIDATES = [16, 32, 64, 128]  # cpp 批量档位（串行 bs=1 仅 py 后端，已随 py 降级移除）


@pytest.fixture(autouse=True)
def _enable_info_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    yield


def _install_uniform_policy(monkeypatch):
    """均匀先验 + 零价值 mock（统一推理入口 calc_batch_arrays）。"""
    import numpy as np

    def fake_calc_batch_arrays(model, boards, *args, **kwargs):
        boards = np.asarray(boards)
        n = boards.shape[0]
        policies = np.zeros((n, 225), dtype=np.float64)
        for i in range(n):
            empty = boards[i] == 0
            cnt = int(empty.sum())
            if cnt > 0:
                policies[i][empty] = 1.0 / cnt
        return np.zeros(n, dtype=np.float64), policies

    monkeypatch.setattr(mcts_mod, "calc_batch_arrays", fake_calc_batch_arrays)


def _win_board():
    """1 有横向活四（成五点为 (7,2)/(7,7)）的均势残局。"""
    board = new_board()
    for j in range(3, 7):
        board[7][j] = 1
    for j in range(0, 8, 2):
        board[0][j] = -1
    return board


def _quality(model_factory, batch_size, num_simulations):
    """返回 (是否找到致胜点, 致胜点访问占比, 实际落子)。

    生产 cpp 后端（stub 拦截统一推理入口），树状态经 dump_state() 读取。
    """
    import numpy as np
    model = model_factory()
    mcts = MCTS(model, c_puct=1.0, noise_eps=0.0,
                infer_batch_size=batch_size)
    (_v, probs), root_id = mcts.run(
        copy.deepcopy(_win_board()), num_simulations, return_root=True)
    move = calc_next_move(_win_board(), probs, temperature=0)

    state = mcts.dump_state()
    win_visits = 0
    total = 0
    ch = state['children'][root_id]
    for a in np.flatnonzero(ch >= 0):
        n = int(state['visits'][int(ch[a])])
        total += n
        r, c = divmod(int(a), board_size)
        if (r, c) in [(7, 2), (7, 7)]:
            win_visits += n
    share = win_visits / max(total, 1)
    return move in [(7, 2), (7, 7)], share, move


def _timing(model, batch_size, sims):
    """同进程单局自对弈（生产 cpp 后端），返回 (秒/手, 手数)。"""
    t0 = time.perf_counter()
    game = generate_single_game(
        0, model.state_dict(), sims,
        model.hidden_channels, model.num_blocks,
        infer_batch_size=batch_size,
        opening_eval_batch_size=Config.opening_eval_batch_size)
    elapsed = time.perf_counter() - t0
    moves = len(game[0])
    return elapsed / moves, moves


def test_batch_size_sweep(monkeypatch):
    sims = Config.train_simulation

    def model_factory():
        return ValueCNN(hidden_channels=8, num_blocks=2).eval()

    # 速度扫描用真实大模型（不打 mock 补丁）
    real_model = ValueCNN().to(get_device()).eval()
    _timing(real_model, 32, sims=5)  # 预热

    logger.info("=" * 78)
    logger.info("infer_batch_size 扫描（%d 模拟/步, 设备 %s）", sims, get_device())

    results = {}
    for bs in CANDIDATES:
        sec_per_move, moves = _timing(real_model, bs, sims)
        # 质量检查在临时补丁上下文中做（不影响真实网络计时局）
        with monkeypatch.context() as m:
            _install_uniform_policy(m)
            ok100, share100, mv100 = _quality(model_factory, bs, 100)
            ok200, share200, mv200 = _quality(model_factory, bs, 200)
        results[bs] = dict(sec=sec_per_move, ok100=ok100, ok200=ok200,
                           share100=share100, share200=share200, moves=moves)

    # 质量判定为绝对标准：200 模拟必须命中致胜点（100 模拟仅记录参考）
    logger.info("-" * 78)
    logger.info("  %6s | %-12s | %-22s | %-12s | %s",
                "batch", "秒/手", "致胜点命中(100/200模拟)",
                "致胜点访问占比", "评价")
    logger.info("-" * 78)
    for bs in CANDIDATES:
        r = results[bs]
        logger.info("  %6d | %8.3f    | %s / %s   | %5.1f%% / %5.1f%%     | %s [%d 手]",
                    bs, r["sec"],
                    "✓" if r["ok100"] else "✗",
                    "✓" if r["ok200"] else "✗",
                    r["share100"] * 100, r["share200"] * 100,
                    "质量达标" if r["ok200"] else "质量不达标!", r["moves"])
    logger.info("-" * 78)

    for bs in CANDIDATES:
        assert results[bs]["ok200"], f"batch={bs} 200 模拟未命中致胜点"

    # 推荐：质量达标中最快的
    best = min(CANDIDATES, key=lambda bs: results[bs]["sec"])
    logger.info("推荐 infer_batch_size = %d（质量达标，%.3f 秒/手）",
                best, results[best]["sec"])
    logger.info("=" * 78)
