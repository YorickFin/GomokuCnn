"""验证 MCTS 每步都在根节点重新注入 Dirichlet 噪声。

修复前：树复用后新根已展开，_run_batched 跳过根展开块，
        Dirichlet 噪声不会重新注入，根 priors 沿用子节点的干净先验。
修复后：run() 在树复用后显式调用 _reapply_root_noise，
        从 clean_priors 重新混合噪声，每步根节点都有新鲜探索噪声。

测试方法：
1. 用均匀先验 + 零价值的假网络隔离搜索机制；
2. patch np.random.dirichlet 记录调用次数；
3. 经 dump_state() 读取根节点 priors / clean_priors，验证加噪形态
   （py/cpp 同构导出；生产为 cpp 后端）。
"""

import numpy as np
import pytest

from src.board import flip_perspective, new_board
from src.config import board_size
from src.mcts import MCTS, calc_next_move
from src.net import ValueCNN


def make_model():
    return ValueCNN(hidden_channels=8, num_blocks=2).eval()


def install_uniform_policy(monkeypatch):
    """均匀先验 + 零价值，消除网络随机性。

    py/cpp 两后端的推理都走 src.mcts.calc_batch_arrays 统一路径，
    patch 该符号即可覆盖单盘与批量入口。
    """
    def fake_calc_batch_arrays(model, boards, device, channels_last=False,
                               fp16=False, profile=None):
        n = boards.shape[0]
        b = boards.reshape(n, board_size, board_size)
        values = np.zeros(n)
        policies = np.zeros((n, board_size * board_size))
        for i in range(n):
            empty = (b[i] == 0).reshape(-1)
            if empty.any():
                policies[i][empty] = 1.0 / int(empty.sum())
        return values, policies

    monkeypatch.setattr("src.mcts.calc_batch_arrays",
                        fake_calc_batch_arrays)


def _root_priors_uniform(mcts, board, attr="priors"):
    """判断根节点 priors(或 clean_priors) 是否等于均匀分布。"""
    legal = (np.asarray(board) == 0).ravel()
    p = mcts.dump_state()[attr][0][legal]
    uniform = 1.0 / legal.sum()
    return np.allclose(p, uniform, atol=1e-12)


def _spy_dirichlet(monkeypatch, call_log):
    real_dirichlet = np.random.dirichlet

    def spy(alpha, size=None):
        call_log.append(len(alpha))
        return real_dirichlet(alpha, size=size)

    monkeypatch.setattr(np.random, "dirichlet", spy)


@pytest.mark.parametrize("batch_size", [8])
def test_dirichlet_reapplied_on_every_move(monkeypatch, batch_size):
    """第一手和树复用后的每一手，根节点都应重新注入 Dirichlet 噪声。"""
    install_uniform_policy(monkeypatch)

    dirichlet_calls = []
    _spy_dirichlet(monkeypatch, dirichlet_calls)

    model = make_model()
    mcts = MCTS(model, c_puct=1.0, noise_eps=0.25, dirichlet_alpha=0.1,
                infer_batch_size=batch_size)
    board = new_board()

    # ---- 第 1 手：全新树（cur_root=None） ----
    dirichlet_calls.clear()
    (_, probs1), root1 = mcts.run(board, num_simulations=50,
                                  return_root=True, add_noise=True)
    print(f"\n[第 1 手] batch_size={batch_size}")
    print(f"  dirichlet 调用次数: {len(dirichlet_calls)}")
    print(f"  priors 是否均匀(未加噪): {_root_priors_uniform(mcts, board)}")
    print(f"  clean_priors 是否均匀: "
          f"{_root_priors_uniform(mcts, board, 'clean_priors')}")
    assert len(dirichlet_calls) == 1, "第一手应注入一次 Dirichlet 噪声"
    assert not _root_priors_uniform(mcts, board), \
        "第一手根 priors 因噪声应当偏离均匀分布"
    assert _root_priors_uniform(mcts, board, "clean_priors"), \
        "clean_priors 应保持干净（均匀），未被噪声污染"

    # 落第 1 手
    action = calc_next_move(board, probs1, temperature=1.0)
    board[action[0]][action[1]] = 1
    flip_perspective(board)

    # ---- 第 2 手：树复用（cur_root=上一手选中的子节点） ----
    child_root = mcts.next_root(root1, action)
    assert child_root is not None

    dirichlet_calls.clear()
    (_, probs2), root2 = mcts.run(board, num_simulations=50,
                                  cur_root=child_root,
                                  return_root=True, add_noise=True)
    print(f"[第 2 手] batch_size={batch_size} (树复用)")
    print(f"  dirichlet 调用次数: {len(dirichlet_calls)}")
    print(f"  priors 是否均匀(未加噪): {_root_priors_uniform(mcts, board)}")
    print(f"  clean_priors 是否均匀: "
          f"{_root_priors_uniform(mcts, board, 'clean_priors')}")
    assert len(dirichlet_calls) == 1, \
        "树复用后应重新注入一次 Dirichlet 噪声"
    assert not _root_priors_uniform(mcts, board), \
        "树复用后根 priors 因噪声应当偏离均匀分布"
    assert _root_priors_uniform(mcts, board, "clean_priors"), \
        "clean_priors 应保持干净，不得被二次加噪污染"

    # ---- 第 3 手：再次树复用 ----
    action2 = calc_next_move(board, probs2, temperature=1.0)
    board[action2[0]][action2[1]] = 1
    flip_perspective(board)
    child_root2 = mcts.next_root(root2, action2)
    assert child_root2 is not None

    dirichlet_calls.clear()
    mcts.run(board, num_simulations=50, cur_root=child_root2,
             add_noise=True)
    print(f"[第 3 手] batch_size={batch_size} (树复用)")
    print(f"  dirichlet 调用次数: {len(dirichlet_calls)}")
    assert len(dirichlet_calls) == 1, \
        "第三步树复用后仍应重新注入 Dirichlet 噪声"
    assert not _root_priors_uniform(mcts, board), \
        "第三步根 priors 因噪声应当偏离均匀分布"


def test_no_noise_when_add_noise_false(monkeypatch):
    """add_noise=False 时，即使树复用也不应注入噪声（eval 场景）。"""
    install_uniform_policy(monkeypatch)

    dirichlet_calls = []
    _spy_dirichlet(monkeypatch, dirichlet_calls)

    model = make_model()
    mcts = MCTS(model, c_puct=1.0, noise_eps=0.25, dirichlet_alpha=0.1,
                infer_batch_size=8)
    board = new_board()

    # 第 1 手：add_noise=False
    dirichlet_calls.clear()
    (_, probs1), root1 = mcts.run(board, num_simulations=50,
                                  return_root=True, add_noise=False)
    assert len(dirichlet_calls) == 0
    assert _root_priors_uniform(mcts, board), \
        "add_noise=False 时根 priors 应保持干净均匀"

    # 树复用后仍 add_noise=False
    action = calc_next_move(board, probs1, temperature=0)
    board[action[0]][action[1]] = 1
    flip_perspective(board)
    child_root = mcts.next_root(root1, action)

    dirichlet_calls.clear()
    mcts.run(board, num_simulations=50, cur_root=child_root,
             add_noise=False)
    assert len(dirichlet_calls) == 0, "eval 场景不应注入噪声"
    assert _root_priors_uniform(mcts, board), \
        "add_noise=False 且树复用时根 priors 应保持干净均匀"


def test_clean_priors_not_double_noised(monkeypatch):
    """连续多步后 clean_priors 仍等于干净先验，未被二次加噪。

    这验证 _reapply_root_noise 从 clean_priors 重新混合，
    而非在已加噪的 priors 上叠加。
    """
    install_uniform_policy(monkeypatch)
    _spy_dirichlet(monkeypatch, [])

    model = make_model()
    mcts = MCTS(model, c_puct=1.0, noise_eps=0.25, dirichlet_alpha=0.1,
                infer_batch_size=8)
    board = new_board()

    (_, probs), root = mcts.run(board, num_simulations=40,
                                return_root=True, add_noise=True)
    action = calc_next_move(board, probs, temperature=1.0)
    board[action[0]][action[1]] = 1
    flip_perspective(board)
    child = mcts.next_root(root, action)

    mcts.run(board, num_simulations=40, cur_root=child, add_noise=True)

    # clean_priors 应精确等于均匀分布（网络先验），priors 应偏离
    state = mcts.dump_state()
    legal = (np.asarray(board) == 0).ravel()
    clean = state['clean_priors'][0][legal]
    noisy = state['priors'][0][legal]
    uniform = 1.0 / legal.sum()

    assert np.allclose(clean, uniform, atol=1e-12), \
        "clean_priors 必须严格等于干净网络先验，不得累积噪声"
    assert not np.allclose(noisy, uniform, atol=1e-3), \
        "priors 必须因噪声偏离均匀分布"


if __name__ == "__main__":
    from unittest.mock import patch

    class _MP:
        def __init__(self):
            self._patches = []

        def setattr(self, target, val):
            patcher = patch(target, val)
            patcher.start()
            self._patches.append(patcher)

        def undo(self):
            for p in reversed(self._patches):
                p.stop()

    mp = _MP()
    try:
        test_dirichlet_reapplied_on_every_move(mp, batch_size=1)
        test_dirichlet_reapplied_on_every_move(mp, batch_size=8)
        test_no_noise_when_add_noise_false(mp)
        test_clean_priors_not_double_noised(mp)
        print("\n所有检查通过。")
    finally:
        mp.undo()
