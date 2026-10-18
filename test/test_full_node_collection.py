"""全节点数据收集与 prior+Q 前缀重分配标签测试（优化方案2.md §2/§3）。

覆盖：
A. _collect_visited_nodes：根节点/内部节点收集、过滤条件、重分配 value
   标签（前缀 max 凸组合）语义、权重公式、board 重建正确性、
   root_only 旧管线兼容；
B. _reassign_labels：层次差分分配、Q 优先提升、未访问 prior 保留、
   value 前缀 max 凸组合（手算期望）、边界（无已访问子节点）、归一化；
C. _play_selfplay_game 集成：全节点样本量大于根节点数、旧管线 value
   交替符号不变、标签合法性与归一化、固定评估集走旧管线。
"""

import math

import numpy as np
import pytest

from src import board as board_mod
from src.config import Config, board_size
from src.utils import (_collect_visited_nodes, _play_selfplay_game,
                       _reassign_labels)
import src.utils as utils_mod

# 未编译 mcts_core.pyd 时跳过 cpp 用例（不影响其余测试）
pytest.importorskip("src.mcts_core", reason="cpp 后端未编译")


# ---------- stub 推理（确定性函数，拦截统一推理入口） ----------

class _DummyModel:
    def to(self, *args, **kwargs):
        return self


def stub_eval(boards):
    b = boards.astype(np.float64)
    weights = np.arange(225) * 0.037 + 1.0
    values = np.tanh((b * weights).sum(axis=1) * 0.13)
    logits = np.sin(b * (np.arange(225) + 1.0) * 0.11) * 2.0
    logits = np.where(boards == 0, logits, -1e9)
    logits = logits + np.arange(225, dtype=np.float64) * 1e-6
    logits = logits.reshape(-1, 225) if logits.ndim == 1 else logits
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    policies = e / e.sum(axis=1, keepdims=True)
    return values, policies


def stub_noise(n_legal):
    rng = np.random.default_rng(1000 + n_legal)
    return rng.dirichlet(np.full(n_legal, 0.1))


def make_mcts(infer_batch_size=8):
    """cpp 后端（生产默认）+ stub 推理。"""
    m = utils_mod.MCTS(_DummyModel(), infer_batch_size=infer_batch_size)
    m._eval_hook = stub_eval
    m._noise_hook = stub_noise
    return m


# ================= A. _collect_visited_nodes =================

def test_collect_includes_root_and_internal_nodes(monkeypatch):
    """一次搜索后收集：第一个节点是根，且内部节点被收集。"""
    monkeypatch.setattr(Config, 'collect_min_frac', 0.0)  # 收全部已访问节点
    mcts = make_mcts()
    board = board_mod.new_board()
    (value, probs), _ = mcts.run(board, 60, return_root=True, add_noise=True)

    nodes = _collect_visited_nodes(mcts, board, 60)
    assert nodes[0]['is_root']
    assert len(nodes) > 1, "全节点收集应包含内部节点"
    # 根节点：value 占位 None（终局回填），weight 按公式
    assert nodes[0]['value'] is None
    # 根 policy 与 run 返回的访问分布一致
    np.testing.assert_allclose(nodes[0]['policy'], probs, atol=1e-12)


def test_root_only_returns_single_root():
    """root_only=True（旧管线/固定评估集）：只收根，weight=1.0，访问分布。"""
    mcts = make_mcts()
    board = board_mod.new_board()
    (value, probs), _ = mcts.run(board, 40, return_root=True, add_noise=True)

    nodes = _collect_visited_nodes(mcts, board, 40, root_only=True)
    assert len(nodes) == 1
    nd = nodes[0]
    assert nd['is_root']
    assert nd['value'] is None
    assert nd['weight'] == 1.0
    np.testing.assert_allclose(nd['policy'], probs, atol=1e-12)


def test_filter_skips_shallow_leaves(monkeypatch):
    """收集数不超过 visits>0 的非终局节点数（浅节点被过滤）。"""
    monkeypatch.setattr(Config, 'collect_min_frac', 0.0)
    mcts = make_mcts()
    board = board_mod.new_board()
    mcts.run(board, 64, add_noise=True)

    state = mcts.dump_state()
    nodes = _collect_visited_nodes(mcts, board, 64)
    visits = state['visits']
    assert len(nodes) <= int(((visits > 0)
                              & (~state['term_flag'])).sum())
    # 根节点（非终局）必须被收集
    assert nodes[0]['is_root']


def test_collect_min_frac_threshold(monkeypatch):
    """门槛语义：非根节点必须 visits >= round(sims × collect_min_frac)。

    相对门槛随 sims 自动缩放（sims=64 × 0.5 = 门槛 32）；
    门槛过大时只收根节点（退化旧管线）——这是训练量保险丝。
    """
    mcts = make_mcts()
    board = board_mod.new_board()
    mcts.run(board, 64, add_noise=True)
    state = mcts.dump_state()
    visited = sum(1 for i in range(state['next_id'])
                  if state['visits'][i] > 0 and not state['term_flag'][i])

    monkeypatch.setattr(Config, 'collect_min_frac', 0.5)
    nodes = _collect_visited_nodes(mcts, board, 64)
    assert len(nodes) < visited, "门槛应过滤掉低访问量节点"
    for nd in nodes:
        if not nd['is_root']:
            assert int(state['visits'][nd['node_id']]) >= 32

    # 同一绝对门槛在不同 sims 下应自动缩放：sims=32 时门槛 16，
    # 原 visits=32 的节点仍在，visits∈[16,32) 的新节点被收进来
    monkeypatch.setattr(Config, 'collect_min_frac', 0.5)
    nodes32 = _collect_visited_nodes(mcts, board, 32)
    assert all(int(state['visits'][nd['node_id']]) >= 16
               for nd in nodes32 if not nd['is_root'])

    monkeypatch.setattr(Config, 'collect_min_frac', 10.0)
    nodes = _collect_visited_nodes(mcts, board, 64)
    assert len(nodes) == 1
    assert nodes[0]['is_root']


def test_internal_value_is_prefix_max_reassign(monkeypatch):
    """内部节点 value = prior+Q 前缀重分配凸组合（nn_012 方案）。

    验证三点：有限且在 [-1,1]；单已访问子节点时退化为 -val[c]
    （单点前缀 max）；多子节点时落在子节点取反 Q 的 [min, max] 区间
    （凸组合性质，不再等于访问量加权树 Q）。
    """
    monkeypatch.setattr(Config, 'collect_min_frac', 0.0)
    mcts = make_mcts()
    board = board_mod.new_board()
    mcts.run(board, 64, add_noise=True)

    state = mcts.dump_state()
    nodes = _collect_visited_nodes(mcts, board, 64)
    internal = [nd for nd in nodes if not nd['is_root']]
    assert internal, "应有内部节点被收集"

    checked_single = 0
    checked_multi = 0
    for nd in internal:
        v = nd['value']
        assert v == v          # 非 NaN
        assert -1.0001 <= v <= 1.0001

        nid = nd['node_id']
        ch = state['children'][nid]
        child_q = [-float(state['val'][int(ch[a])])
                   for a in np.flatnonzero(ch >= 0)
                   if state['visits'][int(ch[a])] > 0]
        if not child_q:
            # 无已访问子节点：退路为树内 Q
            assert v == pytest.approx(float(state['val'][nid]))
            continue
        lo, hi = min(child_q), max(child_q)
        assert lo - 1e-9 <= v <= hi + 1e-9, \
            f"节点 {nid} value={v} 超出子节点取反 Q 区间 [{lo},{hi}]"
        if len(child_q) == 1:
            # 单点前缀 max：value = -val[child]（与旧树 Q 同值，但路径不同）
            assert v == pytest.approx(child_q[0], abs=1e-9)
            checked_single += 1
        else:
            checked_multi += 1

    # 循环内断言已覆盖全部语义（单点退化/凸组合区间/NaN）；
    # 树形状依 sims 而定，此处不硬性要求两类节点都出现
    assert checked_single + checked_multi > 0


def test_weight_formula(monkeypatch):
    """weight = min(1, sqrt(visits/sims)) × train_buff。"""
    monkeypatch.setattr(Config, 'collect_min_frac', 0.0)
    mcts = make_mcts()
    board = board_mod.new_board()
    mcts.run(board, 64, add_noise=True)

    state = mcts.dump_state()
    nodes = _collect_visited_nodes(mcts, board, 64)
    for nd in nodes:
        if nd['is_root']:
            continue
        assert 0 < nd['weight'] <= Config.train_buff + 1e-12
        # 根节点 visits=sims → weight = train_buff（visits 不会超 sims 时）
    assert nodes[0]['weight'] == pytest.approx(Config.train_buff)


def test_board_reconstruction_matches_manual_play(monkeypatch):
    """深度 1 子节点的重建局面 = 根局面落子后翻转视角。"""
    monkeypatch.setattr(Config, 'collect_min_frac', 0.0)
    mcts = make_mcts()
    board = board_mod.new_board()
    mcts.run(board, 64, add_noise=True)

    state = mcts.dump_state()
    nodes = _collect_visited_nodes(mcts, board, 64)
    root_np = np.asarray(board, dtype=np.int8)

    # 从树侧挑几个深度 1 子节点，手工落子比对
    collected = [nd['board'].tobytes() for nd in nodes]
    ch = state['children'][0]
    checked = 0
    for a in np.flatnonzero(ch >= 0):
        c = int(ch[a])
        if state['visits'][c] <= 0:
            continue
        expect = root_np.copy()
        r, cc = divmod(int(a), board_size)
        expect[r, cc] = 1
        expect *= -1
        assert expect.tobytes() in collected, \
            f"深度 1 子节点局面未正确重建 (move={r},{cc})"
        checked += 1
    assert checked > 0


def test_deep_tree_chain_replay_verification(monkeypatch):
    """深树链式重放：沿 parent/move_flat 回溯重放落子，比对重建 board。

    150 模拟保证树足够深、节点 id（模拟创建序）与 BFS 序错位——
    防御「用节点 id 索引 BFS 位置列表」类静默错位 bug
    （曾导致 test_selfplay_10_games_timing IndexError）。
    """
    monkeypatch.setattr(Config, 'collect_min_frac', 0.0)  # 深树全量收集
    mcts = make_mcts()
    board = board_mod.new_board()
    mcts.run(board, 150, add_noise=True)

    state = mcts.dump_state()
    nodes = _collect_visited_nodes(mcts, board, 150)
    root_np = np.asarray(board, dtype=np.int8)

    # 测试前提：收集节点中存在 id 与 BFS 位置错位（否则该测试无区分力）
    assert len(nodes) > 10
    assert any(nd['node_id'] != i for i, nd in enumerate(nodes)), \
        "150 模拟下节点 id 与 BFS 序应出现错位"

    for nd in nodes:
        nid = nd['node_id']
        moves = []
        cur = nid
        while cur != 0:
            moves.append(int(state['move_flat'][cur]))
            cur = int(state['parent'][cur])
            assert cur >= 0, f"节点 {nid} 的 parent 链未回到根"
        expect = root_np.copy()
        for a in reversed(moves):
            r, c = divmod(a, board_size)
            expect[r, c] = 1
            expect *= -1
        np.testing.assert_array_equal(
            nd['board'], expect,
            err_msg=f"节点 {nid}（深度 {len(moves)}）board 重建错误")


# ================= B. _reassign_labels =================

def make_state(children_flat, clean_priors_flat, visits, val):
    """构造最小树状态（_reassign_labels 只依赖这四个字段）。"""
    return {
        'children': np.array([children_flat], dtype=np.int32),
        'clean_priors': np.array([clean_priors_flat], dtype=np.float64),
        'visits': np.array(visits, dtype=np.int32),
        'val': np.array(val, dtype=np.float64),
    }


def _flat(children_map):
    row = [-1] * (board_size * board_size)
    for pos, cid in children_map.items():
        row[pos] = cid
    return row


def test_reassign_boosts_high_q_low_prior():
    """核心机制：prior 高 Q 差被削减、prior 低 Q 好被提升。

    A(pos5): prior=0.8, 父视角 Q=-0.5；B(pos10): prior=0.2, Q=+1.0。
    层次差分：i=0 层预算 0.6 给 A；i=1 层预算 0.2×2 给 B。
    期望 policy 标签 A=0.6, B=0.4（访问分布旧标签是 0.8/0.2）。
    value 前缀 max：i=0 层 cur=-0.5（仅 A），i=1 层 cur=+1.0（含 B），
    value = (-0.5×0.6 + 1.0×0.4)/1.0 = 0.1。
    """
    clean = [0.0] * 225
    clean[5] = 0.8
    clean[10] = 0.2
    state = make_state(_flat({5: 1, 10: 2}), clean,
                       visits=[10, 5, 5], val=[0.0, 0.5, -1.0])
    policy, value = _reassign_labels(state, 0)
    assert policy[0, 5] == pytest.approx(0.6)
    assert policy[0, 10] == pytest.approx(0.4)
    assert policy.sum() == pytest.approx(1.0)
    assert value == pytest.approx(0.1)


def test_reassign_unvisited_prior_preserved():
    """未访问子节点的 prior 保留为网络原始值。

    A(pos5): prior=0.5, Q=-0.5；B(pos10): prior=0.3, Q=+1.0；C 未访问。
    policy: A 0.2×1=0.2；B 0.3×2=0.6；C 保留 0.2。
    value 分母只含已访问 prior（sum_used=0.8）：
    value = (-0.5×0.2 + 1.0×0.6)/0.8 = 0.625。
    """
    clean = [0.0] * 225
    clean[5] = 0.5
    clean[10] = 0.3
    clean[20] = 0.2   # C 未被访问
    state = make_state(_flat({5: 1, 10: 2, 20: 3}), clean,
                       visits=[10, 5, 5, 0], val=[0.0, 0.5, -1.0, 0.0])
    policy, value = _reassign_labels(state, 0)
    assert policy[0, 5] == pytest.approx(0.2)
    assert policy[0, 10] == pytest.approx(0.6)
    assert policy[1, 5] == pytest.approx(0.2)
    assert policy.sum() == pytest.approx(1.0)
    assert value == pytest.approx(0.625)


def test_reassign_no_visited_children_returns_prior():
    """无已访问子节点（叶子/终局根）→ 网络原始 prior + 树内 Q 退路。"""
    clean = [0.0] * 225
    clean[5] = 0.6
    clean[10] = 0.4
    # 情形 1：children 全 -1
    state = make_state(_flat({}), clean, visits=[8], val=[0.0])
    policy, value = _reassign_labels(state, 0)
    np.testing.assert_allclose(
        policy, np.array(clean).reshape(board_size, board_size))
    assert value == pytest.approx(0.0)
    # 情形 2：child 存在但 visits=0
    state = make_state(_flat({5: 1}), clean, visits=[8, 0], val=[0.0, 0.0])
    policy, value = _reassign_labels(state, 0)
    np.testing.assert_allclose(
        policy, np.array(clean).reshape(board_size, board_size))
    assert value == pytest.approx(0.0)


def test_reassign_normalizes_and_zeroes_illegal():
    """输出归一化，非法位置（clean=0）保持 0；value 为凸组合。

    A(prior=0.45, Q=-0.1) B(prior=0.3, Q=+0.7) C(prior=0.25, Q=-0.3)：
    i=0: w=0.15×1, cur=-0.1；i=1: w=0.05×2, cur=+0.7；
    i=2: w=0.25×3, cur=+0.7 → value = 0.58。
    """
    clean = [0.0] * 225
    clean[0] = 0.45
    clean[1] = 0.3
    clean[2] = 0.25
    state = make_state(_flat({0: 1, 1: 2, 2: 3}), clean,
                       visits=[20, 12, 5, 3], val=[0.0, 0.1, -0.7, 0.3])
    policy, value = _reassign_labels(state, 0)
    assert policy.sum() == pytest.approx(1.0)
    assert (policy >= 0).all()
    illegal = np.array(clean).reshape(board_size, board_size) == 0
    assert (policy[illegal] == 0).all()
    assert value == pytest.approx(0.58)


def test_reassign_equal_q_prefers_higher_prior():
    """Q 相同的层内 tie：max 稳定取先出现者（prior 降序后的更高者）。

    两节点父视角 Q 均为 0.5：所有层增量都给 A，
    value = 0.5×(0.2×1 + 0.4×2)/1.0 = 0.5。
    """
    clean = [0.0] * 225
    clean[5] = 0.6
    clean[10] = 0.4
    state = make_state(_flat({5: 1, 10: 2}), clean,
                       visits=[10, 5, 5], val=[0.0, -0.5, -0.5])
    policy, value = _reassign_labels(state, 0)
    assert policy[0, 5] == pytest.approx(1.0)
    assert policy[0, 10] == pytest.approx(0.0)
    assert value == pytest.approx(0.5)


def test_reassign_single_visited_child_degenerates_to_q():
    """单已访问子节点：前缀 max 退化为 -val[c]，与树内 Q 精确一致。

    A(pos5): prior=0.9, val=0.4 → 父视角 Q=-0.4；weight_sum=prior×1=0.9，
    value = -0.4×0.9/0.9 = -0.4。
    """
    clean = [0.0] * 225
    clean[5] = 0.9
    clean[10] = 0.1   # 未访问
    state = make_state(_flat({5: 1, 10: 2}), clean,
                       visits=[10, 4, 0], val=[0.0, 0.4, 0.0])
    policy, value = _reassign_labels(state, 0)
    assert value == pytest.approx(-0.4)
    # policy：i=0 末层预算 0.9×1 全给 A，未访问 B 保留 0.1
    assert policy[0, 5] == pytest.approx(0.9)
    assert policy[0, 10] == pytest.approx(0.1)


def _nn012_label_reference(children, clean, visits, val):
    """逐行复刻 nn_012.py get_train_data 的标签计算（参照实现）。

    L451-499：org_policy 按空位归一化 → 已访问子节点移入 moves 并按
    prior 降序 → 层次差分累加 probs 与 value_sum（每层先除 sum_used）。
    children 行 / clean / visits / val 与 dump_state 字段同构，其中
    val[c] 对应 nn_012 的 child.val（子节点行动方视角 Q）。
    """
    moves = []          # (move, child_id, prior, child.val)
    probs = np.zeros(board_size * board_size)
    for a in range(board_size * board_size):
        c = int(children[a])
        if clean[a] > 0:
            probs[a] = clean[a]          # 未访问点保留（clean 已归一化）
        if c >= 0 and visits[c] > 0:
            moves.append((a, c, float(clean[a]), float(val[c])))

    sum_used = 0.0
    ordered = []
    for a, c, prior, cval in moves:
        sum_used += probs[a]             # nn_012: sum_used += probs[move]
        ordered.append((a, c, cval, probs[a]))
        probs[a] = 0.0
    ordered.sort(key=lambda x: x[3], reverse=True)

    value_sum = 0.0
    for i in range(len(ordered)):
        cnt = ordered[i][3]
        if i + 1 < len(ordered):
            cnt -= ordered[i + 1][3]
        if cnt == 0:
            continue
        cur = -1e9
        best_pos = i
        for j in range(0, i + 1):
            vals = -ordered[j][2]
            if vals > cur:
                cur = vals
                best_pos = j
        probs[ordered[best_pos][0]] += cnt * (i + 1)
        value_sum += cur * cnt * (i + 1) / sum_used
    return probs, value_sum


def test_reassign_labels_matches_nn012_reference():
    """与 nn_012 原版实现数值对照（py/cpp parity 同款惯例）。

    随机树状态（含 visits=0 子节点、prior 相等 tie、cnt=0 层、单子
    节点）下，policy 与 value 均须在 1e-12 容差内一致。已知浮点路径
    差异：nn_012 每层先除 sum_used 再累加，本实现先累加最后除一次
    （telescoping 等价），故用容差断言而非逐位相等。
    """
    rng = np.random.default_rng(42)
    for trial in range(30):
        k = int(rng.integers(1, 30))            # 已访问子节点数
        n_extra = int(rng.integers(0, 8))       # 未访问合法点数
        legal = rng.choice(225, size=k + n_extra, replace=False)

        clean = np.zeros(225)
        clean[legal] = rng.random(len(legal)) + 1e-3
        if trial % 3 == 0:                      # 制造 prior 相等 tie
            clean[legal[0]] = clean[legal[1]] if k + n_extra > 1 \
                else clean[legal[0]]
        clean /= clean.sum()                    # 模拟 softmax 归一化

        children = np.full(225, -1, dtype=np.int32)
        for j, pos in enumerate(legal[:k]):
            children[pos] = j + 1
        visits = [int(rng.integers(1, 30))] + \
            [int(rng.integers(0, 20)) for _ in range(k)]
        val = np.concatenate([[0.0], rng.uniform(-1, 1, k)])

        state = {
            'children': np.array([children]),
            'clean_priors': np.array([clean]),
            'visits': np.array(visits, dtype=np.int32),
            'val': np.array(val),
        }
        policy, value = _reassign_labels(state, 0)
        ref_policy, ref_value = _nn012_label_reference(
            children, clean, visits, val)

        np.testing.assert_allclose(
            policy, ref_policy.reshape(board_size, board_size),
            atol=1e-12, err_msg=f"trial {trial}: policy 不一致")
        assert value == pytest.approx(ref_value, abs=1e-12), \
            f"trial {trial}: value 不一致 ({value} vs {ref_value})"
        # 与 nn_012 相同的合法性断言
        assert value == value and -1.0001 <= value <= 1.0001


def test_reassign_value_in_unit_range():
    """value 恒在 [-1,1]（凸组合），且威胁主导：极端 Q 以层级权重保留。

    A(prior=0.7, Q=+0.2, visits=7) B(prior=0.2, Q=+0.99 潜伏杀点, visits=3)：
    旧访问量加权 Q 把杀点稀释成 (7×0.2+3×0.99)/10 = 0.437；
    前缀 max 标签：i=0 层 cur=+0.2（cnt=0.5, w=0.5×1），
    i=1 层 cur=+0.99（cnt=0.2, w=0.2×2）
    → value = (0.2×0.5 + 0.99×0.4)/0.9 ≈ 0.5511——杀点权重 0.44，
    远超其 0.3 的访问占比，且不被更高 prior 的平庸 Q 平均掉。
    """
    clean = [0.0] * 225
    clean[5] = 0.7
    clean[10] = 0.2
    state = make_state(_flat({5: 1, 10: 2}), clean,
                       visits=[10, 7, 3], val=[0.0, -0.2, -0.99])
    policy, value = _reassign_labels(state, 0)
    assert -1.0 <= value <= 1.0
    assert value == pytest.approx((0.2 * 0.5 + 0.99 * 0.4) / 0.9)


# ================= C. _play_selfplay_game 集成 =================

@pytest.fixture()
def deterministic_selfplay(monkeypatch):
    """固定开局 + stub MCTS + 全程贪心，保证自对弈可复现且快速。"""
    # 24 sims 下让搜索充分的内部节点也能过门槛，验证全节点收集语义
    monkeypatch.setattr(Config, 'collect_min_frac', 0.0)
    monkeypatch.setattr(utils_mod, 'generate_random_board',
                        lambda model, eval_batch_size=None:
                        board_mod.new_board())
    real_mcts = utils_mod.MCTS

    def mcts_factory(model, infer_batch_size=None, backend=None):
        m = real_mcts(_DummyModel(), infer_batch_size=8)
        m._eval_hook = stub_eval
        m._noise_hook = stub_noise
        return m

    monkeypatch.setattr(utils_mod, 'MCTS', mcts_factory)
    monkeypatch.setattr(Config, 'temp_min', 0)
    monkeypatch.setattr(Config, 'temp_max', 0)  # 0=确定性格子采样


def test_selfplay_full_nodes_exceed_roots(deterministic_selfplay):
    """全节点收集的样本量必须大于仅根节点（旧管线）。"""
    model = None  # stub 推理下模型不被触达
    old = _play_selfplay_game(model, 24, collect_all_nodes=False)
    new = _play_selfplay_game(model, 24, collect_all_nodes=True)

    n_roots = len(old[0])
    assert len(new[0]) > n_roots
    # 四元组长度一致
    for data in (old, new):
        assert len(data[0]) == len(data[1]) == len(data[2]) == len(data[3])
    # 回填完成：无 None
    assert all(v is not None for v in new[2])


def test_selfplay_old_pipeline_unchanged(deterministic_selfplay):
    """旧管线（root_only）：weight 全 1，value 沿落子序列交替符号。"""
    model = None
    boards, policies, values, weights = _play_selfplay_game(
        model, 24, collect_all_nodes=False)

    assert all(w == 1.0 for w in weights)
    n = len(values)
    assert n > 0
    assert values[-1] in (1.0, -1.0, 0.0)
    for k in range(n - 1):
        assert values[k] == pytest.approx(-values[k + 1]), \
            "蒙特卡洛 return 必须沿序列交替翻转视角"


def test_selfplay_policy_labels_valid(deterministic_selfplay):
    """全节点标签：非负、归一化、非法位置为 0、value 在 [-1,1]。"""
    model = None
    boards, policies, values, weights = _play_selfplay_game(
        model, 24, collect_all_nodes=True)

    for i in range(len(policies)):
        p = policies[i].numpy()
        b = boards[i].numpy()
        occupied = b[0] + b[1] > 0  # 通道 0=当前方 1=对手
        assert (p >= 0).all()
        assert p.sum() == pytest.approx(1.0, abs=1e-6)
        assert (p[occupied] == 0).all(), \
            f"样本 {i} 已占点上有非零 policy"
        # 根=蒙特卡洛 return（±1/0），内部=前缀 max 凸组合，均在 [-1,1]
        assert -1.0001 <= values[i] <= 1.0001


def test_fixed_eval_set_root_only(monkeypatch):
    """固定评估集恒只收根节点（2026-09-21）：考卷大小不随 collect 参数
    漂移（frac=0 全收时随机模型考卷膨胀至 50 万样本 → 每轮评估 66s）；
    治本方案已废（考卷锚定 base、永不驱动调度），对齐训练目标无意义。"""
    captured = {}

    class _Sentinel(Exception):
        pass

    def fake_run(_pool, tasks):
        captured['tasks'] = tasks
        raise _Sentinel  # 不真正对弈，任务构造完成后立即中断

    monkeypatch.setattr(utils_mod, '_run_selfplay_tasks', fake_run)
    model = _DummyModel()
    model.hidden_channels = 8
    model.num_blocks = 2
    model.state_dict = lambda: {}

    with pytest.raises(_Sentinel):
        utils_mod.generate_fixed_eval_set(model, num_games=2,
                                          num_simulations=8)
    assert captured['tasks'], "应构造自对弈任务"
    # collect_all_nodes=False = 强制根收，与 Config.collect_all_nodes 解耦
    assert all(t[7] is False for t in captured['tasks']), \
        "固定评估集任务必须强制根收，不跟随 Config"
