"""自对弈数据生成、随机开局、数据增强与数据集。

职责边界：
- 棋盘规则见 src/board.py；
- 网络见 src/net.py；
- 搜索见 src/mcts.py。

数据管线要点（优化方案2.md）：
1. value 标签：实际走到的根节点用对局最终胜负（+1/-1/0）沿落子序列
   回溯（蒙特卡洛 return）；树中内部节点用树内 Q（自举，无终局对应）；
2. policy 标签：collect_all_nodes=True 时全部收集节点用 prior + Q 值
   层次重分配（纠正 prior 盲区），False 时用根节点访问分布（旧行为）；
3. 数据收集：搜索树中访问量达标的节点（不只根节点），
   过滤 visits >= round(sims × collect_min_frac)（相对门槛，自动随
   train_simulation 缩放），权重 sqrt(visits/sims)；
4. 固定评估集恒用旧管线（只收根节点 + 访问分布标签），保证考卷构成稳定；
5. 先按“局”划分训练/验证集，再做 D4 数据增强，消除对称样本泄漏；
6. 温度随落子步数退火（开局高温探索，后期低温收敛）。
"""

import logging
import math
import multiprocessing
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src import board as board_mod
from src.config import Config, board_size
from src.mcts import MCTS, calc_next_move
from src.net import (ValueCNN, board_to_tensor, get_calc, get_calc_batch,
                     get_device)

logger = logging.getLogger(__name__)


def _reassign_labels(state, nid):
    """prior+Q 前缀重分配：同一分层循环产出 policy 与 value 标签。

    把节点 nid 的网络干净先验（clean_priors）当作"初始概率预算"，
    在已访问子节点间按 prior 降序逐层差分切分，每层增量
    (prior[i]-prior[i+1])×(i+1) 分配给层内（prior 更高的前 i+1 个走法中）
    Q 最高者（前缀 max）：

    - policy：prior 高但 Q 差的走法被削减，prior 低但 Q 好的走法被提升；
      未访问子节点的 prior 保留为网络原始判断。Σcnt×(i+1)=Σ已访问 prior
      （telescoping），质量守恒，输出归一化。
    - value：同一循环累加每层前缀 max 的 Q（当前节点视角），
      value = Σ cur_i×cnt_i×(i+1) / sum_used，权重和恰为 sum_used，
      是凸组合、恒在 [-1,1]。与旧树内 Q（访问量加权平均）的本质差异：
      潜伏威胁点（如对手杀点）只要被访问过一次且 prior 靠前，其极端
      价值就以 prior 层级权重全额主导标签，不被访问占比稀释、不被
      平均化抹平——训练后 value head 可在单次前向识别潜伏威胁，
      对弈时无需深搜索即可避开留杀走法（nn_012 实证）。

    返回 (policy (15,15) 归一化分布, value float)。moves 为空（无已访问
    子节点）时退回 (原始 prior, 节点树内 Q)。

    state: MCTS.dump_state() 的 dict（py/cpp 后端同构）。
    """
    ch = state['children'][nid]
    clean = state['clean_priors'][nid]

    # 收集已访问子节点：(prior, 当前节点视角 Q, flat move)
    moves = []
    for a in np.flatnonzero(ch >= 0):
        c = int(ch[a])
        if state['visits'][c] > 0:
            # child.val 是子节点视角的 Q，取反转为当前节点视角
            moves.append((float(clean[a]), -float(state['val'][c]), int(a)))

    # 边界：无已访问子节点（浅叶子/终局根）→ prior 原样 + 树内 Q
    if not moves:
        return (clean.reshape(board_size, board_size),
                float(state['val'][nid]))

    moves.sort(key=lambda x: x[0], reverse=True)

    policy_flat = np.zeros(board_size * board_size, dtype=np.float64)
    # 未访问子节点（含未创建）的 prior 保留（非法位置 clean=0 天然为 0）
    visited_mask = np.zeros(board_size * board_size, dtype=bool)
    for _, _, a in moves:
        visited_mask[a] = True
    policy_flat[~visited_mask] = clean[~visited_mask]

    # 层次差分分配：第 i 层预算 = prior[i] - prior[i+1]（末层保留全部）；
    # value 同步累加该层前缀 max 的 Q × 同一权重（与 nn_012 同构）
    n = len(moves)
    value_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        cnt = moves[i][0]
        if i + 1 < n:
            cnt -= moves[i + 1][0]
        if cnt <= 0:
            continue
        # 在 prior 更高的前 i+1 个走法中选 Q 最高者
        best = max(moves[:i + 1], key=lambda x: x[1])
        policy_flat[best[2]] += cnt * (i + 1)
        value_sum += best[1] * cnt * (i + 1)
        weight_sum += cnt * (i + 1)

    total = policy_flat.sum()
    if total <= 0:
        return (clean.reshape(board_size, board_size),
                float(state['val'][nid]))
    policy = (policy_flat / total).reshape(board_size, board_size)
    # weight_sum=0（全部 prior 相等的退化情形）时按树内 Q 退路
    value = value_sum / weight_sum if weight_sum > 0 \
        else float(state['val'][nid])
    return policy, value


def _collect_visited_nodes(mcts, root_board, num_sims, root_only=False):
    """一次搜索完成后，从搜索树收集训练节点（优化方案2.md §3）。

    过滤：visits >= round(sims × Config.collect_min_frac)（相对门槛）——
    只收集搜索
    充分的节点，跳过浅分叉叶子（Q 不可靠、权重 0.04 纯噪声；原
    child_count>1 过滤过宽致数据 ~230x）。root_only=True 时只
    收集根节点（旧管线 / 固定评估集，考卷构成保持不变）。

    返回 list of dict：
      board: (15,15) int8 numpy（1=当前行动方，视角归一）
      policy: (15,15) float64 标签分布（重分配或访问分布）
      value: 根节点为 None（终局后回填蒙特卡洛 return）；
             内部节点在 policy_reassign=True 时为 prior+Q 前缀重分配
             凸组合（nn_012 方案），否则为树内 Q（节点视角平均价值）
      weight: min(1, sqrt(visits/num_sims)) × train_buff（root_only 时 1.0；
              min 截断是因为树复用下 visits 含跨手累计，可能超本手预算）
      is_root: 是否本手根节点
    """
    state = mcts.dump_state()
    visits = state['visits']
    root_np = np.asarray(root_board, dtype=np.int8)

    # BFS 全树，重建每个节点局面：子局面 = 父局面落子(1) 后翻转视角。
    # 与搜索内工作棋盘约定一致（mcts._descend 同款操作）。
    # 注意 boards/order 按 BFS 位置索引，与树节点 id 是两个空间
    # （树 id 按模拟创建序分配，与 BFS 序不同），经由 order[pos] 映射。
    boards = [root_np]
    order = [0]
    pos = 0
    while pos < len(order):
        nid = order[pos]
        row = state['children'][nid]
        for a in np.flatnonzero(row >= 0):
            b = boards[pos].copy()
            r, cc = divmod(int(a), board_size)
            b[r, cc] = 1
            b *= -1
            boards.append(b)
            order.append(int(row[a]))
        pos += 1

    nodes = []
    for idx, nid in enumerate(order):
        nvis = int(visits[nid])
        if nvis <= 0:
            continue
        if state['term_flag'][nid]:
            # 终局节点（五连/满盘）无合法着法集合，policy 标签无定义；
            # 其价值信息由父节点与根节点蒙特卡洛 return 承载，跳过。
            continue
        ch = state['children'][nid]
        cvis = np.where(ch >= 0, visits[np.maximum(ch, 0)], 0)
        is_root = idx == 0
        if root_only and not is_root:
            continue
        # 收集门槛（相对比例，随 train_simulation 自动缩放）：仅保留
        # visits >= round(sims × collect_min_frac) 的节点。原 child_count > 1
        # 过滤太宽松（visits=2 浅分叉占大头，权重 0.04 噪声主导），训练数据
        # ~230x 导致单轮训练 24~49 小时。
        min_vis = round(num_sims * Config.collect_min_frac)
        if not is_root and nvis < min_vis:
            continue

        if is_root or not Config.policy_reassign:
            # 访问分布（旧标签；根节点与 MCTS._get_results 同公式）
            total = cvis.sum()
            if total > 0:
                policy = (cvis / total).reshape(board_size, board_size)
            else:
                policy = state['clean_priors'][nid].reshape(
                    board_size, board_size)
            value = None if is_root else float(state['val'][nid])
        else:
            # prior+Q 前缀重分配：policy 与 value 由同一分层循环产出
            #（nn_012 方案；value 为前缀 max 凸组合，非访问量加权 Q）
            policy, value = _reassign_labels(state, nid)
        if root_only:
            weight = 1.0
        else:
            weight = (min(1.0, math.sqrt(nvis / num_sims))
                      * Config.train_buff)

        nodes.append({
            'node_id': nid,
            'board': boards[idx],
            'policy': policy,
            'value': value,
            'weight': weight,
            'is_root': is_root,
        })
    return nodes


def generate_random_board(model, eval_batch_size=None):
    """生成随机的均势开局棋盘。

    开局分布：
    1. 以 opening_empty_prob 概率直接从空盘开始——空盘由 MCTS 自行落子，
       当前模型强度下天然集中在中心常规开局；
    2. 其余随机预落 0~opening_max_moves 手，候选落子全盘均匀（不限制中心，
       保留非常规开局覆盖）。

    随机落子后做视角归一化，保证返回棋盘时“1 = 当前行动方”；
    并在若干候选中选取网络估值绝对值最小（最均势）的局面。
    候选局面分批批量推理（eval_batch_size<=1 时退化为逐个推理）。
    """
    if eval_batch_size is None:
        eval_batch_size = Config.opening_eval_batch_size
    device = get_device()
    model = model.to(device)

    all_positions = [(i, j) for i in range(board_size)
                     for j in range(board_size)]

    best_board = board_mod.new_board()
    if random.random() < Config.opening_empty_prob:
        num_run = 0  # opening_empty_prob 概率直接从空盘开始
    else:
        num_run = random.randint(200, 1000)

    candidates = []
    for _ in range(num_run):
        num_moves = random.randint(0, Config.opening_max_moves)
        board = board_mod.new_board()
        current_player = 1  # 1 先手，保证视角约定一致
        # 每个候选独立采样 num_moves 个不同位置，避免所有候选共享
        # 同一个 perm 前缀导致多样性不足（实际唯一开局数 = max_moves+1）。
        for i, j in random.sample(all_positions, num_moves):
            board[i][j] = current_player
            current_player = -current_player
        if board_mod.evaluation_func(board) != 0:
            continue
        # 视角归一化：落子结束后下一手应由 1 行动；若轮到 -1 则整体翻转
        if current_player == -1:
            board_mod.flip_perspective(board)
        candidates.append(board)

    best_val = 1e9

    def _consider(board, value):
        nonlocal best_val, best_board
        val = max(float(value), -float(value))
        if val < best_val:
            best_val = val
            best_board = board

    if eval_batch_size and eval_batch_size > 1:
        for start in range(0, len(candidates), eval_batch_size):
            batch = candidates[start:start + eval_batch_size]
            for board, (value, _policy) in zip(
                    batch, get_calc_batch(model, batch)):
                _consider(board, value)
    else:
        for board in candidates:
            value, _ = get_calc(model, board)
            _consider(board, value)

    return best_board


def augment_data(boards, policies, values, weights):
    """D4 对称增强（4 次旋转 x 2 次翻转），样本数扩大 8 倍。

    向量化实现：8 个视图各用一次整批 rot90/flip（替代逐样本 Python
    双循环，~60k 样本下 16.8s → 0.7s）。输出与旧逐样本实现逐位一致
    （视图顺序 k 外层 flip 内层，每样本 8 视图连续；values/weights
    repeat_interleave 对齐同一排布），由 test_vec_augment_parity 验证。
    """
    b_views, p_views = [], []
    for k in range(4):
        b = torch.rot90(boards, k, [2, 3])     # (N,3,H,W) 平面维=[2,3]
        p = torch.rot90(policies, k, [1, 2])   # (N,H,W) 平面维=[1,2]
        b_views += [b, torch.flip(b, [3])]     # W 翻转对齐单样本 flip([2])
        p_views += [p, torch.flip(p, [2])]
    n = boards.shape[0]
    return (
        torch.stack(b_views, dim=1).reshape(
            n * 8, *boards.shape[1:]).contiguous(),
        torch.stack(p_views, dim=1).reshape(
            n * 8, *policies.shape[1:]).contiguous(),
        values.repeat_interleave(8, dim=0),
        weights.repeat_interleave(8, dim=0),
    )


def apply_d4(board, policy, k, flip):
    """对单个样本施加 D4 群中的一种对称变换（与 augment_data 同一套）。

    board: (3, H, W)，policy: (H, W)；k ∈ {0,1,2,3} 为 90° 旋转次数，
    flip 为是否再做水平翻转。board 与 policy 必须同步变换。
    """
    new_board = torch.rot90(board, k, [1, 2])
    new_policy = torch.rot90(policy, k, [0, 1])
    if flip:
        new_board = torch.flip(new_board, [2])
        new_policy = torch.flip(new_policy, [1])
    return new_board, new_policy


class WeightedDataset(Dataset):
    def __init__(self, boards, policies, values, weights, augment=False):
        """augment=False：原样返回（用于已确定性增强的验证/固定评估集）；
        augment=True：每次 __getitem__ 随机抽一种 D4 变换在线增强，
        不物化 8 份副本，同一样本在不同 epoch 看到不同视图。
        """
        self.boards = boards
        self.policies = policies
        self.values = values
        self.weights = weights
        self.augment = augment

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        policy = self.policies[idx]
        if self.augment:
            k = random.randrange(4)
            flip = random.random() < 0.5
            board, policy = apply_d4(board, policy, k, flip)
        return board, policy, self.values[idx], self.weights[idx]


def _play_selfplay_game(model, num_simulations, infer_batch_size=None,
                        opening_eval_batch_size=None, collect_all_nodes=None,
                        backend=None):
    """用已构建的模型自对弈一局，返回单局 (boards, policies, values, weights)。

    抽离自 generate_single_game：持久进程池的 worker 复用同一模型连下
    多局，避免每局重建模型（原实现每局都 ValueCNN + load_state_dict）。

    collect_all_nodes（None=读 Config.collect_all_nodes）：
    True 时收集搜索树中所有访问过的节点（value：根=蒙特卡洛 return、
    内部=树内 Q）；False 时只收根节点（旧管线，固定评估集用）。
    backend（None=读 Config.mcts_backend）：串行旧路径（infer_batch_size=1）
    必须传 'py'，cpp 后端仅支持批量搜索。
    """
    if collect_all_nodes is None:
        collect_all_nodes = Config.collect_all_nodes

    board = generate_random_board(model, eval_batch_size=opening_eval_batch_size)
    mcts = MCTS(model, infer_batch_size=infer_batch_size, backend=backend)

    # 每局随机采样温度（nn_012 方案：0.1~0.8，整局统一）
    game_temp = random.uniform(Config.temp_min, Config.temp_max)

    state_boards = []
    state_policies = []
    state_values = []      # 根节点先存 None 占位，终局后回填 return
    state_weights = []
    played_root_idx = []   # 实际走到的根节点在样本列表中的索引（按落子序）
    root = None

    with torch.no_grad():
        for move_num in range(board_size * board_size):
            (value, action_probs), new_root = mcts.run(
                board, num_simulations, cur_root=root,
                return_root=True, add_noise=True)

            # 收集训练节点（含本手根节点；root_only=旧管线只收根）
            for nd in _collect_visited_nodes(
                    mcts, board, num_simulations,
                    root_only=not collect_all_nodes):
                if nd['is_root']:
                    played_root_idx.append(len(state_values))
                state_boards.append(board_to_tensor(nd['board']))
                state_policies.append(torch.FloatTensor(nd['policy']))
                state_values.append(nd['value'])
                state_weights.append(nd['weight'])

            temperature = game_temp
            action = calc_next_move(board, action_probs, temperature)
            board[action[0]][action[1]] = 1

            # 树复用：把实际落子对应的子树作为下一手的根
            root = mcts.next_root(new_root, action)

            if mcts.is_terminal(board):
                break

            # 翻转视角，交由对手行动
            board_mod.flip_perspective(board)

    # 终局结果：break 时 board 未翻转，1 为最后落子方
    final_eval = board_mod.evaluation_func(board)
    if final_eval > 0:
        game_result = 1.0   # 最后落子方获胜
    elif final_eval < 0:
        game_result = -1.0
    else:
        game_result = 0.0   # 下满棋盘，平局

    # 根节点回填蒙特卡洛 return：沿实际落子序列交替翻转视角
    n_played = len(played_root_idx)
    for k, idx in enumerate(played_root_idx):
        state_values[idx] = game_result * ((-1) ** (n_played - 1 - k))

    return (
        torch.stack(state_boards),
        torch.stack(state_policies),
        state_values,
        state_weights,
    )


def generate_single_game(_, model_state_dict, num_simulations,
                         hidden_channels, num_blocks,
                         infer_batch_size=None, opening_eval_batch_size=None,
                         collect_all_nodes=None, backend=None):
    """生成单局自对弈数据（独立构建模型）。

    保留原签名供单局调用/测试；多进程训练管线走 _selfplay_chunk_worker，
    worker 内模型只构建一次、连下多局。
    """
    model = ValueCNN(hidden_channels=hidden_channels, num_blocks=num_blocks)
    model.load_state_dict(model_state_dict)
    model.eval()
    return _play_selfplay_game(
        model, num_simulations,
        infer_batch_size=infer_batch_size,
        opening_eval_batch_size=opening_eval_batch_size,
        collect_all_nodes=collect_all_nodes, backend=backend)


def _selfplay_chunk_worker(task):
    """进程池 worker：构建一次模型，连下 n_games 局。

    task = (n_games, model_state_dict, num_simulations, hidden_channels,
            num_blocks, infer_batch_size, opening_eval_batch_size,
            collect_all_nodes, backend)
    返回 list[game_tuple]。state_dict 为 CPU tensor，子进程加载后 .to(device)。
    """
    (n_games, model_state_dict, num_simulations, hidden_channels,
     num_blocks, infer_batch_size, opening_eval_batch_size,
     collect_all_nodes, backend) = task

    model = ValueCNN(hidden_channels=hidden_channels, num_blocks=num_blocks)
    model.load_state_dict(model_state_dict)
    model.eval()

    games = []
    for _ in range(n_games):
        games.append(_play_selfplay_game(
            model, num_simulations,
            infer_batch_size=infer_batch_size,
            opening_eval_batch_size=opening_eval_batch_size,
            collect_all_nodes=collect_all_nodes, backend=backend))
    return games


def _split_balanced(total, n_parts):
    """把 total 均分成 n_parts 份，返回每份大小的列表。"""
    base = total // n_parts
    rem = total % n_parts
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def _run_selfplay_tasks(pool, tasks):
    """把按块任务派发给进程池，汇总所有局结果。"""
    games = []
    for chunk_games in pool.imap_unordered(_selfplay_chunk_worker, tasks):
        games.extend(chunk_games)
    return games


def concat_games(games):
    """将多局数据拼接为 (boards, policies, values, weights)。"""
    boards = torch.cat([g[0] for g in games], dim=0)
    policies = torch.cat([g[1] for g in games], dim=0)
    values = torch.FloatTensor([v for g in games for v in g[2]])
    weights = torch.FloatTensor([w for g in games for w in g[3]])
    return boards, policies, values, weights


def split_games(games, train_ratio):
    """按局随机划分训练/验证集，返回 (train_games, val_games)。"""
    games = list(games)
    random.shuffle(games)
    num_train = int(len(games) * train_ratio)
    return games[:num_train], games[num_train:]


def _build_selfplay_tasks(model, num_games, num_simulations, n_workers,
                          infer_batch_size, opening_eval_batch_size,
                          collect_all_nodes=None, backend=None):
    """构造按块自对弈任务（每个 worker 一块、块内连下多局）。"""
    # state_dict 转 CPU 再传给子进程，避免 CUDA tensor IPC 共享告警
    model_state_dict = {k: v.detach().cpu()
                        for k, v in model.state_dict().items()}
    chunk_sizes = [s for s in _split_balanced(num_games, n_workers) if s > 0]
    return [
        (size, model_state_dict, num_simulations,
         model.hidden_channels, model.num_blocks,
         infer_batch_size, opening_eval_batch_size, collect_all_nodes,
         backend)
        for size in chunk_sizes
    ]


def _generate_games(model, num_games, num_simulations=None,
                    infer_batch_size=None, opening_eval_batch_size=None,
                    pool=None, collect_all_nodes=None, backend=None):
    """多进程自对弈生成 num_games 局原始棋局（未切分、未增强）。

    pool 给定时复用外部持久进程池（调用方负责关闭）；否则临时建池、
    用完即关。返回 list of (boards, policies, values, weights)，每局一组。
    collect_all_nodes=None 时读 Config.collect_all_nodes；
    backend=None 时读 Config.mcts_backend（infer_batch_size=1 必须传 'py'）。
    """
    if num_simulations is None:
        num_simulations = Config.train_simulation

    n_workers = max(1, min(
        pool._processes if pool is not None else Config.num_workers,
        num_games))
    tasks = _build_selfplay_tasks(
        model, num_games, num_simulations, n_workers,
        infer_batch_size, opening_eval_batch_size, collect_all_nodes,
        backend)

    if pool is not None:
        return _run_selfplay_tasks(pool, tasks)

    with multiprocessing.get_context('spawn').Pool(
            processes=len(tasks)) as p:
        return _run_selfplay_tasks(p, tasks)


def generate_selfplay_data(model, num_games, num_simulations=None,
                           infer_batch_size=None, opening_eval_batch_size=None,
                           pool=None, backend=None):
    """多进程生成自对弈数据，按局切分并做 D4 增强。

    返回 (train_data, val_data)，每个为
    (boards, policies, values, weights) 堆叠 tensor。
    infer_batch_size=1 / opening_eval_batch_size=1 可回退到逐个推理旧路径
    （此时必须 backend='py'）；pool 给定时复用外部持久进程池。
    """
    games = _generate_games(
        model, num_games, num_simulations=num_simulations,
        infer_batch_size=infer_batch_size,
        opening_eval_batch_size=opening_eval_batch_size,
        pool=pool, backend=backend)

    train_games, val_games = split_games(games, Config.train_ratio)
    train_data = augment_data(*concat_games(train_games))
    val_data = augment_data(*concat_games(val_games))

    logger.info("自对弈 %d 局：训练 %d 局 / 验证 %d 局；"
                "增强后训练样本 %d，验证样本 %d",
                len(games), len(train_games), len(val_games),
                len(train_data[0]), len(val_data[0]))
    return train_data, val_data


def generate_fixed_eval_set(model, num_games=None, num_simulations=None,
                            pool=None):
    """生成固定评估集：num_games 局自对弈全部用于评估，D4 增强后返回。

    与训练数据完全隔离；落盘后跨轮次复用（见 save/load_fixed_eval_set），
    保证每轮 val loss 在同一份数据上可比。
    恒只收根节点（2026-09-21 起，collect_all_nodes=False 强制）：考卷
    大小不随 collect 参数漂移——frac=0 全收时随机初始化模型的考卷会
    膨胀到 50 万样本（每轮评估 66s）。"考卷对齐训练目标"的治本方案
    已废（考卷锚定 base、与进步结构性脱钩，永不驱动调度），对齐无意义。
    pool 给定时复用外部持久进程池。
    """
    if num_games is None:
        num_games = Config.eval_fixed_games
    games = _generate_games(model, num_games, num_simulations=num_simulations,
                            pool=pool, collect_all_nodes=False)
    return augment_data(*concat_games(games))


def save_fixed_eval_set(path, data):
    """落盘固定评估集 (boards, policies, values, weights)。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(data, path)


def load_fixed_eval_set(path):
    """加载固定评估集；文件不存在时返回 None。"""
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location='cpu', weights_only=True)
