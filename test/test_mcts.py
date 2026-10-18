"""MCTS 搜索测试：策略合法性、致胜手选择、防守判断、根噪声。

搜索机制测试统一走生产后端（Config.mcts_backend='cpp'）；py 纯
Python 实现仅作为参考保留（src/mcts.py），不再单独测试。
"""

import copy

import numpy as np
import pytest

from src.board import evaluation_func, new_board
from src.config import board_size
from src.mcts import MCTS, calc_next_move
from src.net import ValueCNN


def make_model():
    return ValueCNN(hidden_channels=8, num_blocks=2).eval()


def empty_board():
    return new_board()


def install_uniform_policy(monkeypatch):
    """用均匀先验 + 零价值替换网络推理，隔离测试 MCTS 搜索机制。

    候选按棋盘字典序被确定性探索，终局价值传播后访问会集中到
    致胜/防守要点，从而消除随机网络先验带来的测试抖动。
    py/cpp 两后端的推理都走 src.mcts.calc_batch_arrays 统一路径，
    patch 该符号即可同时覆盖单盘与批量入口。
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


# ---------- MCTS 搜索 ----------

def test_mcts_probs_legal_and_normalized():
    model = make_model()
    mcts = MCTS(model, noise_eps=0.0)
    board = empty_board()
    board[7][7] = 1
    board[7][8] = -1
    value, probs = mcts.run(board, num_simulations=20)

    assert probs.shape == (board_size, board_size)
    # 已占位置概率为 0
    assert probs[7, 7] == 0.0
    assert probs[7, 8] == 0.0
    # 合法位置概率和为 1
    assert abs(probs.sum() - 1.0) < 1e-6
    assert -1.0 <= value <= 1.0


def test_mcts_finds_immediate_win(monkeypatch):
    """1 有横向活四，MCTS 必须选择成五的点。"""
    install_uniform_policy(monkeypatch)
    model = make_model()
    mcts = MCTS(model, c_puct=1.0, noise_eps=0.0)
    board = empty_board()
    # 活四：第 7 行 3..6 列，两端 (7,2) 与 (7,7) 皆空
    for j in range(3, 7):
        board[7][j] = 1
    # 对手 4 个散子（远离、不连成五），保持子数视角合法
    for j in range(0, 8, 2):
        board[0][j] = -1

    _, probs = mcts.run(board, num_simulations=200)
    move = calc_next_move(board, probs, temperature=0)
    assert move in [(7, 2), (7, 7)], f"未选择致胜点，实际选择 {move}"


@pytest.mark.parametrize("infer_batch_size", [32])
def test_mcts_defense_against_open_four(monkeypatch, infer_batch_size):
    """对手冲四（唯一成五点 (7,7)）时的防守判断。

    1 走闲手后翻转视角，对手 MCTS 必须能直接找到成五点 (7,7)；
    1 堵住 (7,7) 后，对手的最优应手落子后不应形成五连。
    （注：对手活四两端皆空时为必杀，不存在有效防守，故此处用冲四。）
    """
    import copy
    install_uniform_policy(monkeypatch)
    model = make_model()
    mcts = MCTS(model, c_puct=1.0, noise_eps=0.0,
                infer_batch_size=infer_batch_size)

    def after_move(board, move):
        """1 在 move 落子后翻转视角，返回对手行动的盘面。"""
        b = copy.deepcopy(board)
        b[move[0]][move[1]] = 1
        for x in range(board_size):
            for y in range(board_size):
                b[x][y] *= -1
        return b

    # 构造对手（-1）冲四：(7,3..6) 为 -1，(7,2) 已被 1 堵住，唯一致胜点 (7,7)
    board = empty_board()
    for j in range(3, 7):
        board[7][j] = -1
    board[7][2] = 1
    for j in range(0, 6, 2):  # 1 的 3 个散子，凑成双方各 4 子
        board[0][j] = 1

    # 闲手 (0,8)：对手视角下 (7,7) 成五即胜
    lose_board = after_move(board, (0, 8))
    _, probs_lose = mcts.run(lose_board, num_simulations=200)
    kill_move = calc_next_move(lose_board, probs_lose, temperature=0)
    assert kill_move == (7, 7), f"对手未找到成五点，实际 {kill_move}"

    # 防守 (7,7)：对手最优应手落子后不能形成五连
    safe_board = after_move(board, (7, 7))
    _, probs_safe = mcts.run(safe_board, num_simulations=200)
    move = calc_next_move(safe_board, probs_safe, temperature=0)
    check = copy.deepcopy(safe_board)
    check[move[0]][move[1]] = 1
    assert evaluation_func(check) == 0, (
        f"防守后对手仍可于 {move} 成五，防守无效")


def test_dirichlet_noise_root_priors_valid():
    """根节点注入 Dirichlet 噪声后先验仍合法（非负、归一化）。

    树状态通过 dump_state() 读取（py/cpp 同构导出，生产为 cpp 后端）。
    """
    import numpy as np
    model = make_model()
    board = empty_board()

    mcts_noise = MCTS(model, noise_eps=0.25, dirichlet_alpha=0.1)
    mcts_noise.run(board, num_simulations=1, add_noise=True)
    # 根节点先验行（空盘全部 225 个着法合法）
    priors = mcts_noise.dump_state()['priors'][0]
    assert abs(priors.sum() - 1.0) < 1e-6
    assert np.all(priors >= 0)
    assert np.all(priors <= 1.0)


def test_batched_search_virtual_loss_cleanup(monkeypatch):
    """批量搜索结束后虚拟访问必须完全撤销：访问计数自洽、无残留状态。"""
    import numpy as np
    install_uniform_policy(monkeypatch)
    model = make_model()
    board = empty_board()
    board[7][7] = 1
    board[7][8] = -1

    mcts_b = MCTS(model, noise_eps=0.0, infer_batch_size=8)
    (_v, _p), root_id = mcts_b.run(
        board, num_simulations=50, return_root=True)

    state = mcts_b.dump_state()
    # root 访问总数 = 模拟次数（虚拟计数已全部撤销）
    assert int(state['visits'][root_id]) == 50
    # 除首次（root 自身作为叶子被评估）外，每次模拟访问一个子节点
    ch = state['children'][root_id]
    child_visits = np.where(ch >= 0, state['visits'][np.maximum(ch, 0)],
                            0).sum()
    assert int(child_visits) == 49
    # val 与 w_sum/visits 自洽（虚拟访问撤销干净）
    assert abs(state['val'][root_id]
               - state['w_sum'][root_id] / state['visits'][root_id]) < 1e-9

    # 全树虚拟计数全部归零
    assert int(state['virt'].sum()) == 0


def test_calc_next_move_greedy():
    import numpy as np
    board = empty_board()
    probs = np.zeros((board_size, board_size))
    probs[3][4] = 0.5
    probs[5][6] = 0.5
    move = calc_next_move(board, probs, temperature=0)
    assert move in [(3, 4), (5, 6)]
    # 已占位置不会被选
    board[3][4] = 1
    move = calc_next_move(board, probs, temperature=0)
    assert move == (5, 6)
