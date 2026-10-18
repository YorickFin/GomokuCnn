"""蒙特卡洛树搜索（MCTS）。

职责边界：
- 棋盘规则（终局判定、视角翻转、落子）见 src/board.py；
- 网络与推理见 src/net.py；
- 自对弈数据生成见 src/utils.py；
- 模型对弈评估与对局封装见 src/eval.py。

搜索要点：
1. 仅根节点注入 Dirichlet 探索噪声；
2. policy 标签使用根节点访问次数分布；
3. 终局节点价值直接来自规则判定，叶子节点价值来自网络；
4. 叶子评估支持批量推理（infer_batch_size > 1）：并发模拟通过
   虚拟访问（virtual count）分散到不同分支，攒够一批叶子后一次
   forward 评估，大幅摊薄 batch=1 推理的 kernel 启动开销。

数据结构（数组化树）：
树节点不使用 Python 对象，而是按节点 id 索引的扁平 numpy 数组：
children(n, 225) 为各着法的子节点 id（-1 未创建），priors/visits/
w_sum 等同理。PUCT 选择对 225 个候选整体向量化 argmax，模拟下行
时用一份可变异的工作棋盘落子/翻转，避免逐节点 deepcopy 棋盘。
跨手树复用通过子树压缩（_reroot）保留统计。
"""

import numpy as np
import torch

from src import board as board_mod
from src.config import Config, board_size
from src.net import (calc_batch_arrays, get_device, load_model,
                     prepare_inference_model)

import time

# C++ 核心（src/mcts_core.pyd，见 build_mcts_core.py）；未编译时退回 Python 后端
try:
    from src import mcts_core as _mcts_core
except ImportError:
    _mcts_core = None

# 15x15 棋盘共 225 个着法，flat index = row * board_size + col
N_MOVES = board_size * board_size


class MCTS:
    def __init__(self, model, c_puct=None, dirichlet_alpha=None, noise_eps=None,
                 infer_batch_size=None, backend=None, channels_last=None,
                 fp16=None):
        self.c_puct = Config.c_puct if c_puct is None else c_puct
        self.dirichlet_alpha = (Config.dirichlet_alpha if dirichlet_alpha is None
                                else dirichlet_alpha)
        self.noise_eps = Config.noise_eps if noise_eps is None else noise_eps
        self.infer_batch_size = (Config.infer_batch_size if infer_batch_size is None
                                 else infer_batch_size)
        # 计算侧推理优化（阶段 2）：默认读 Config，可逐实例覆盖
        # （竞技场新旧路径对弈用）。仅影响本实例的推理路径。
        self.use_channels_last = (Config.inference_channels_last
                                  if channels_last is None else channels_last)
        self.use_fp16 = Config.inference_fp16 if fp16 is None else fp16
        self.device = get_device()
        if isinstance(model, str):
            self.model = load_model(model, self.device)
        else:
            self.model = model.to(self.device)
        if self.use_channels_last:
            self.model = prepare_inference_model(self.model, True)

        # ---- 后端选择 ----
        # 'py'  = 纯 Python 树操作（原实现，支持 infer_batch_size=1 串行路径）
        # 'cpp' = C++ 树操作（src/mcts_core.pyd），网络推理仍在本类中回调
        # 对外 API（run / next_root / is_terminal）完全一致，调用方无感。
        backend = Config.mcts_backend if backend is None else backend
        self.backend = backend
        if backend == 'cpp':
            if _mcts_core is None:
                raise RuntimeError(
                    "mcts_core 未编译：请先运行 python build_mcts_core.py，"
                    "或将 Config.mcts_backend 设为 'py'")
            if not self.infer_batch_size or self.infer_batch_size <= 1:
                raise ValueError(
                    "cpp 后端仅支持批量搜索路径（infer_batch_size > 1）")
            self._cpp = _mcts_core.MctsTree(self.c_puct, self.dirichlet_alpha,
                                            self.noise_eps)
        # 测试钩子：非 None 时替换默认的网络推理 / 噪声采样（用于一致性校验）
        self._eval_hook = None
        self._noise_hook = None
        self._eval_fn = None
        # eval 分段计时累计器（阶段 0-3.1 诊断）：[编码, H2D, forward, D2H, 批数]
        self._eval_profile = [0.0, 0.0, 0.0, 0.0, 0]

    # ---- 基本规则 ----
    def is_terminal(self, board):
        return board_mod.is_full(board) or board_mod.evaluation_func(board) != 0

    # ---- 树存储（扁平数组）----
    def _init_tree(self, cap):
        """分配一棵容量为 cap 个节点的空树，节点 0 为根。"""
        self._cap = cap
        self.children = np.full((cap, N_MOVES), -1, dtype=np.int32)
        self.priors = np.zeros((cap, N_MOVES), dtype=np.float64)
        # 干净先验：未混合 Dirichlet 噪声的网络输出，供树复用后重新加噪
        self.clean_priors = np.zeros((cap, N_MOVES), dtype=np.float64)
        self.visits = np.zeros(cap, dtype=np.int32)
        self.virt = np.zeros(cap, dtype=np.int32)      # 虚拟访问（仅探索项）
        self.w_sum = np.zeros(cap, dtype=np.float64)   # 价值累计
        self.val = np.zeros(cap, dtype=np.float64)     # 节点视角平均价值
        self.value = np.zeros(cap, dtype=np.float64)   # 叶子网络/终局价值
        self.expanded = np.zeros(cap, dtype=bool)
        self.term_flag = np.zeros(cap, dtype=bool)     # 终局节点标记
        self.term_val = np.zeros(cap, dtype=np.float64)
        self.parent = np.full(cap, -1, dtype=np.int32)
        self.move_flat = np.full(cap, -1, dtype=np.int32)
        self.next_id = 1

    def _grow(self, needed):
        """容量不足时翻倍扩容（容量估算兜底，正常路径不触发）。"""
        new_cap = max(self._cap * 2, needed)
        add = new_cap - self._cap
        self.children = np.pad(
            self.children, ((0, add), (0, 0)), constant_values=-1)
        self.priors = np.pad(self.priors, ((0, add), (0, 0)))
        self.clean_priors = np.pad(self.clean_priors, ((0, add), (0, 0)))
        self.visits = np.pad(self.visits, (0, add))
        self.virt = np.pad(self.virt, (0, add))
        self.w_sum = np.pad(self.w_sum, (0, add))
        self.val = np.pad(self.val, (0, add))
        self.value = np.pad(self.value, (0, add))
        self.expanded = np.pad(self.expanded, (0, add))
        self.term_flag = np.pad(self.term_flag, (0, add))
        self.term_val = np.pad(self.term_val, (0, add))
        self.parent = np.pad(self.parent, (0, add), constant_values=-1)
        self.move_flat = np.pad(self.move_flat, (0, add), constant_values=-1)
        self._cap = new_cap

    def _new_node(self, parent_id, move):
        nid = self.next_id
        if nid >= self._cap:
            self._grow(nid + 1)
        self.next_id = nid + 1
        self.parent[nid] = parent_id
        self.move_flat[nid] = move
        return nid

    def _reroot(self, old_root, num_simulations):
        """把 old_root 子树压缩为新数组（根 id 归 0），保留全部统计。

        跨手树复用：上一手选中的子节点成为新根，其可达子树整体
        搬迁到紧凑数组，后续模拟在其上继续。
        容量 = 复用子树节点数 + 本轮新增节点（≤ num_simulations）+ 余量。
        """
        mapping = {old_root: 0}
        order = [old_root]
        for nid in order:  # BFS（order 在迭代中增长）
            for a in np.flatnonzero(self.children[nid] >= 0):
                c = int(self.children[nid, a])
                if c not in mapping:
                    mapping[c] = len(order)
                    order.append(c)

        cap = len(order) + num_simulations + 256
        old_children, old_priors = self.children, self.priors
        old_clean = self.clean_priors
        old_visits, old_virt = self.visits, self.virt
        old_w, old_val = self.w_sum, self.val
        old_value = self.value
        old_exp, old_tf, old_tv = self.expanded, self.term_flag, self.term_val
        self._init_tree(cap)

        for new_id, nid in enumerate(order):
            self.visits[new_id] = old_visits[nid]
            self.w_sum[new_id] = old_w[nid]
            self.val[new_id] = old_val[nid]
            self.value[new_id] = old_value[nid]
            self.expanded[new_id] = old_exp[nid]
            self.term_flag[new_id] = old_tf[nid]
            self.term_val[new_id] = old_tv[nid]
            self.priors[new_id] = old_priors[nid]
            self.clean_priors[new_id] = old_clean[nid]
            for a in np.flatnonzero(old_children[nid] >= 0):
                a = int(a)
                nc = mapping[int(old_children[nid, a])]
                self.children[new_id, a] = nc
                self.parent[nc] = new_id
                self.move_flat[nc] = a
        self.next_id = len(order)

    # ---- 搜索主循环 ----
    def run(self, root_board, num_simulations, cur_root=None,
            return_root=False, add_noise=False):
        if self.backend == 'cpp':
            return self._run_cpp(root_board, num_simulations, cur_root,
                                 return_root, add_noise)

        root_np = np.asarray(root_board, dtype=np.int8)
        if cur_root is not None:
            self._reroot(int(cur_root), num_simulations)
        else:
            # 每次模拟至多新建 1 个节点
            self._init_tree(num_simulations + 256)

        # 树复用后新根已展开，_run_batched/_run_serial 不会再调用
        # _expand(is_root=True)，需显式从干净先验重新注入 Dirichlet 噪声。
        # 全新树时根未展开，噪声由 _expand 负责，此处跳过。
        if add_noise and self.expanded[0]:
            self._reapply_root_noise(0, root_np)

        if self.infer_batch_size and self.infer_batch_size > 1:
            self._run_batched(root_np, num_simulations, add_noise)
        else:
            self._run_serial(root_np, num_simulations, add_noise)

        results = self._get_results(0)
        if return_root:
            return results, 0
        return results

    def next_root(self, root_id, move):
        """落子后返回对应子节点 id（供下一手树复用），未创建则 None。"""
        a = move[0] * board_size + move[1]
        if self.backend == 'cpp':
            ch = int(self._cpp.next_root(int(root_id), a))
            return ch if ch >= 0 else None
        ch = int(self.children[root_id, a])
        return ch if ch >= 0 else None

    def dump_state(self):
        """导出当前树状态（与 C++ dump_state 字段一一对应）。

        供全节点数据收集（优化方案2.md §3）与后端一致性校验使用；
        cpp 后端直接转发 C++ 侧导出，py 后端切片 numpy 数组。
        """
        if self.backend == 'cpp':
            return self._cpp.dump_state()
        n = self.next_id
        return {
            'children': self.children[:n],
            'priors': self.priors[:n],
            'clean_priors': self.clean_priors[:n],
            'visits': self.visits[:n],
            'virt': self.virt[:n],
            'w_sum': self.w_sum[:n],
            'val': self.val[:n],
            'value': self.value[:n],
            'expanded': self.expanded[:n],
            'term_flag': self.term_flag[:n],
            'term_val': self.term_val[:n],
            'parent': self.parent[:n],
            'move_flat': self.move_flat[:n],
            'next_id': self.next_id,
        }

    # ---- C++ 后端 ----
    def _make_eval_fn(self):
        """构建 C++ 侧回调：批量棋盘 (n,225) int8 → (values, policies)。

        走 net.calc_batch_arrays 统一推理路径（py 后端同款），基线数值
        逐位一致；channels_last / fp16 由 Config 或本实例开关决定。
        Config.eval_profile=True 时按批累积四段耗时（阶段 0-3.1 诊断，
        默认关），run() 结束后用 pop_eval_profile() 读取。
        """
        model, device = self.model, self.device

        def eval_fn(boards):
            prof = self._eval_profile if Config.eval_profile else None
            return calc_batch_arrays(
                model, boards, device,
                channels_last=self.use_channels_last, fp16=self.use_fp16,
                profile=prof)

        return eval_fn

    def pop_eval_profile(self):
        """取出并清零 eval 分段计时累计器（阶段 0-3.1 诊断用）。

        返回 dict：encode / h2d / forward / d2h 为秒累计，
        batches 为经历的批次数（含根评估批次）。
        """
        prof = self._eval_profile
        out = {'encode': prof[0], 'h2d': prof[1], 'forward': prof[2],
               'd2h': prof[3], 'batches': int(prof[4])}
        self._eval_profile = [0.0, 0.0, 0.0, 0.0, 0]
        return out

    def _make_noise_fn(self):
        """构建 C++ 侧回调：n 个合法着法的 Dirichlet 噪声（与 _draw_noise 一致）。"""
        alpha = self.dirichlet_alpha

        def noise_fn(n_legal):
            return np.random.dirichlet([alpha] * n_legal)

        return noise_fn

    def _draw_noise(self, n_legal):
        """根节点噪声采样；测试钩子非 None 时替换（默认行为不变）。"""
        if self._noise_hook is not None:
            return self._noise_hook(n_legal)
        return np.random.dirichlet([self.dirichlet_alpha] * n_legal)

    def _run_cpp(self, root_board, num_simulations, cur_root, return_root,
                 add_noise):
        """C++ 后端：树操作全部在 mcts_core 内完成，本类只编排推理回调。"""
        if self._eval_fn is None:
            self._eval_fn = self._make_eval_fn()
        eval_fn = self._eval_hook if self._eval_hook is not None else self._eval_fn
        noise_fn = (self._noise_hook if self._noise_hook is not None
                    else self._make_noise_fn())
        root_np = np.ascontiguousarray(root_board, dtype=np.int8).reshape(-1)
        root_value, probs, root_id = self._cpp.run(
            root_np, int(num_simulations), cur_root, bool(add_noise),
            int(self.infer_batch_size), eval_fn, noise_fn)
        if return_root:
            return (float(root_value), probs), int(root_id)
        return float(root_value), probs

    # ---- 推理入口（测试钩子可替换）----
    def _calc_single(self, board):
        """单盘推理；_eval_hook 非 None 时替换为确定性 stub（一致性校验用）。

        非 hook 路径走 net.calc_batch_arrays 统一推理路径（与 cpp 后端
        同一条 eval 路径），基线数值与原 get_calc 逐位一致。
        """
        if self._eval_hook is not None:
            values, policies = self._eval_hook(
                np.asarray(board, dtype=np.int8).reshape(1, -1))
            return (float(values[0]),
                    policies[0].reshape(board_size, board_size).tolist())
        arr = np.asarray(board, dtype=np.int8).reshape(1, -1)
        values, policies = calc_batch_arrays(
            self.model, arr, self.device,
            channels_last=self.use_channels_last, fp16=self.use_fp16)
        return (float(values[0]),
                policies[0].reshape(board_size, board_size).tolist())

    def _calc_batch(self, boards):
        """批量推理；_eval_hook 非 None 时替换为确定性 stub（一致性校验用）。

        非 hook 路径走 net.calc_batch_arrays 统一推理路径（与 cpp 后端
        同一条 eval 路径），基线数值与原 get_calc_batch 逐位一致。
        """
        if self._eval_hook is not None:
            values, policies = self._eval_hook(
                np.asarray(boards, dtype=np.int8).reshape(len(boards), -1))
            return [(float(values[i]),
                     policies[i].reshape(board_size, board_size).tolist())
                    for i in range(len(boards))]
        arr = np.asarray(boards, dtype=np.int8).reshape(len(boards), -1)
        values, policies = calc_batch_arrays(
            self.model, arr, self.device,
            channels_last=self.use_channels_last, fp16=self.use_fp16)
        return [(float(values[i]),
                 policies[i].reshape(board_size, board_size).tolist())
                for i in range(len(boards))]

    def _run_serial(self, root_np, num_simulations, add_noise):
        """串行搜索：每个叶子立即推理（旧路径，用于对比与回退）。"""
        for _ in range(num_simulations):
            path, work, leaf, is_term = self._descend(root_np)
            if is_term:
                value = self.term_val[leaf]
            else:
                value, policy = self._calc_single(work.tolist())
                self.value[leaf] = value
                self._expand(leaf, policy, work,
                             is_root=(leaf == 0 and add_noise))
            self._backprop(path, value)

    def _run_batched(self, root_np, num_simulations, add_noise):
        """批量搜索：并发模拟经虚拟访问去相关，叶子攒批后一次 forward。"""
        # 根节点必须先评估展开：未展开时所有模拟都会堵在根上无法下行。
        # 等价于串行搜索的第一次模拟（终局根则直接由后续模拟回传）。
        if not self.term_flag[0] and not self.expanded[0]:
            tv = self._terminal_value(root_np)
            if tv is not None:
                self.term_flag[0] = True
                self.term_val[0] = tv
            else:
                value, policy = self._calc_single(root_np.tolist())
                self.value[0] = value
                self._expand(0, policy, root_np, is_root=add_noise)
                self._backprop([0], value)
                num_simulations -= 1

        pending = []  # [(leaf_id, path, board_list)]
        for _ in range(num_simulations):
            path, work, leaf, is_term = self._descend(root_np)
            if is_term:
                # 终局价值立即可得，直接回传
                self._backprop(path, self.term_val[leaf])
                continue

            # 叶子登记并施加虚拟访问（促使同批后续模拟探索别处）
            self._apply_virtual(path)
            pending.append((leaf, path, work.tolist()))

            if len(pending) >= self.infer_batch_size:
                self._evaluate_pending(pending, add_noise)
                pending = []

        if pending:
            self._evaluate_pending(pending, add_noise)

    def _evaluate_pending(self, pending, add_noise):
        """对去重后的叶子批量推理，展开并回传真实价值。"""
        boards = {}
        order = []
        for leaf, _path, board in pending:
            if leaf not in boards:
                boards[leaf] = board
                order.append(leaf)

        results = self._calc_batch([boards[leaf] for leaf in order])
        values = {}
        for leaf, (value, policy) in zip(order, results):
            values[leaf] = value
            self.value[leaf] = value
            self._expand(leaf, policy, boards[leaf],
                         is_root=(leaf == 0 and add_noise))

        for leaf, path, _board in pending:
            self._undo_virtual(path)
            self._backprop(path, values[leaf])

    # ---- 模拟下行 ----
    def _descend(self, root_np):
        """从根选择下行至叶子，返回 (path, 工作棋盘, 叶子id, 是否终局)。"""
        work = root_np.copy()
        node = 0
        path = [0]

        while not self.term_flag[node] and self.expanded[node]:
            a = self._select(node, work)
            r, c = divmod(a, board_size)
            work[r, c] = 1
            work *= -1  # 落子后翻转视角（同 board.play_move）

            ch = int(self.children[node, a])
            if ch < 0:
                ch = self._new_node(node, a)
                self.children[node, a] = ch
            path.append(ch)
            node = ch

            if self.term_flag[node]:
                break
            tv = self._terminal_value(work, (r, c))
            if tv is not None:
                self.term_flag[node] = True
                self.term_val[node] = tv
                break

        return path, work, node, bool(self.term_flag[node])

    @staticmethod
    def _terminal_value(work, last_move=None):
        """终局返回节点视角价值（满盘 0 / 五连 ±衰减分），非终局 None。

        五连只可能由刚落下的一手形成，故只需沿该点的 4 条直线
        数同色连子（O(1)），避免旧实现每次全板扫描 winner()。
        last_move 为 None 时（根盘防御性检查）退回全板判定。
        """
        if not np.any(work == 0):
            return 0.0
        if last_move is None:
            score = board_mod.evaluation_func(work)
            return float(score) if score != 0 else None

        r, c = last_move
        v = work[r, c]  # 落子后已翻转视角：刚落的子为 -1
        for dr, dc in board_mod.DIRECTIONS:
            cnt = 1
            rr, cc = r + dr, c + dc
            while board_mod.in_bounds(rr, cc) and work[rr, cc] == v:
                cnt += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while board_mod.in_bounds(rr, cc) and work[rr, cc] == v:
                cnt += 1
                rr -= dr
                cc -= dc
            if cnt >= 5:
                stones = int(np.count_nonzero(work))
                return float(v) * (1 - stones * 3e-4)
        return None

    def _select(self, node, work):
        """向量化 PUCT：返回最优着法的 flat index。"""
        ch = self.children[node]
        has_child = ch >= 0
        n_visits = np.where(has_child, self.visits[ch], 0)
        n_virt = np.where(has_child, self.virt[ch], 0)
        w_sum = np.where(has_child, self.w_sum[ch], 0.0)

        visited = n_visits > 0
        # 兄弟平均价值仅用真实统计（虚拟访问不含真实价值，不得污染）
        total_real = int(n_visits.sum())
        q_child = w_sum / np.maximum(n_visits, 1)
        ave = w_sum[visited].sum() / (total_real + 1e-5)
        exploit = np.where(visited, q_child, ave)

        explore_buff = np.sqrt(total_real + 1)
        scores = (self.c_puct * self.priors[node] * explore_buff
                  / (n_visits + n_virt + 1) - exploit)
        # 非法着法（已占点）不可选
        scores = np.where(work.ravel() == 0, scores, -1e9)
        return int(np.argmax(scores))

    def _expand(self, node, policy, board, is_root=False):
        """根据网络先验展开节点：先验归一化 + 根节点 Dirichlet 噪声。"""
        legal = (np.asarray(board) == 0).ravel()
        p = np.asarray(policy, dtype=np.float64).reshape(-1) * legal
        total = p.sum()
        if total <= 0:
            p = legal.astype(np.float64) / max(int(legal.sum()), 1)
        else:
            p = p / total

        # 始终保存干净先验，供树复用后根节点重新加噪使用
        self.clean_priors[node] = p

        if is_root and self.noise_eps > 0 and legal.any():
            eta = self._draw_noise(int(legal.sum()))
            p[legal] = ((1 - self.noise_eps) * p[legal]
                        + self.noise_eps * eta)

        self.priors[node] = p
        self.expanded[node] = True

    def _reapply_root_noise(self, node, board):
        """树复用后根节点已展开，从干净先验重新混合 Dirichlet 噪声。

        不能在已加噪的 priors 上二次叠加（会双重加噪导致分布失真），
        因此从 clean_priors 重新混合，visit 统计保持不变。
        """
        if self.noise_eps <= 0:
            return
        legal = (np.asarray(board) == 0).ravel()
        if not legal.any():
            return
        p = self.clean_priors[node].copy()
        eta = self._draw_noise(int(legal.sum()))
        p[legal] = ((1 - self.noise_eps) * p[legal]
                    + self.noise_eps * eta)
        self.priors[node] = p

    # ---- 回传 ----
    def _backprop(self, path, value):
        """沿搜索路径回传价值，逐层翻转视角。"""
        for nid in reversed(path):
            self.visits[nid] += 1
            self.w_sum[nid] += value
            self.val[nid] = self.w_sum[nid] / self.visits[nid]
            value = -value

    def _apply_virtual(self, path):
        """施加虚拟访问：只增 virt（计入探索项分母），不动真实统计。"""
        for nid in path:
            self.virt[nid] += 1

    def _undo_virtual(self, path):
        """撤销虚拟访问（真实价值回传前调用）。"""
        for nid in path:
            self.virt[nid] -= 1

    # ---- 结果 ----
    def _get_results(self, root_id):
        """返回 (根节点价值, 访问次数策略分布)。"""
        ch = self.children[root_id]
        child_visits = np.where(ch >= 0, self.visits[ch], 0).astype(np.float64)
        total = child_visits.sum()
        probs = (child_visits / total).reshape(board_size, board_size) \
            if total > 0 else np.zeros((board_size, board_size))
        root_value = (self.w_sum[root_id] / self.visits[root_id]
                      if self.visits[root_id] > 0 else 0.0)
        return root_value, probs


def calc_next_move(board, probs, temperature=0):
    """根据策略分布选择落子点。temperature=0 为纯贪心。"""
    valid_moves = []
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == 0:
                valid_moves.append((i, j, probs[i][j]))
    if temperature == 0:
        valid_moves.sort(key=lambda x: x[2], reverse=True)
        return valid_moves[0][:2]
    moves = [(i, j) for i, j, _ in valid_moves]
    p = np.array([prob for _, _, prob in valid_moves], dtype=np.float64)
    if temperature != 1.0:
        p = p ** (1.0 / temperature)
    p = p / p.sum()
    chosen = np.random.choice(len(moves), p=p)
    return moves[chosen]
