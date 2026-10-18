"""自对弈 + 训练管线各环节耗时剖析。

分段计时（perf_counter，GPU 段前后 synchronize）：
1. 单局自对弈（串行，真实 100 模拟/步）：
   - 随机开局生成（内含网络推理）
   - MCTS 搜索：拆为「网络推理」与「树选择/扩展/回传（Python）」
   - 落子/编码/树复用等杂项
   - 终局标签与 tensor 整理
   - D4 数据增强（×8）
2. 多进程管线开销：spawn 进程池启动、worker 内模型构建/权重加载；
3. 训练环节：单个 batch（256）forward+backward 耗时，外推每轮训练时间。

属于性能基准而非功能测试，运行约 1~2 分钟：
    .venv\\Scripts\\python.exe -m pytest test/test_benchmark_stages.py -s
"""

import copy
import logging
import multiprocessing as mp
import os
import random
import time

import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from src import mcts as mcts_mod
from src import utils as utils_mod
from src.board import flip_perspective
from src.config import Config
from src.mcts import MCTS
from src.net import ValueCNN, board_to_tensor, get_device
from src.utils import augment_data

logger = logging.getLogger(__name__)


class Timer:
    """累计计时器；GPU 段计时前后同步，保证 CUDA 异步不被漏算。"""

    def __init__(self, sync_cuda=False):
        self.total = 0.0
        self.sync_cuda = sync_cuda

    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.total += time.perf_counter() - self.t0
        return False


@pytest.fixture(autouse=True)
def _enable_info_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    yield


def _install_inference_timer(monkeypatch):
    """包装 MCTS / utils 的批量推理入口，累计推理耗时、批次数与样本数。

    MCTS 统一推理路径为 net.calc_batch_arrays（mcts 模块命名空间导入），
    utils 的随机开局评估仍走 get_calc_batch。
    """
    stats = {
        "mcts_infer_time": 0.0, "mcts_infer_batches": 0,
        "mcts_infer_leaves": 0,
        "open_infer_time": 0.0, "open_infer_batches": 0,
        "open_infer_leaves": 0,
    }

    def make_wrapper(orig, time_key, batch_key, leaf_key):
        def wrapped(model, boards, *args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = orig(model, boards, *args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stats[time_key] += time.perf_counter() - t0
            stats[batch_key] += 1
            stats[leaf_key] += len(boards)
            return result
        return wrapped

    monkeypatch.setattr(
        mcts_mod, "calc_batch_arrays",
        make_wrapper(mcts_mod.calc_batch_arrays, "mcts_infer_time",
                     "mcts_infer_batches", "mcts_infer_leaves"))
    monkeypatch.setattr(
        utils_mod, "get_calc_batch",
        make_wrapper(utils_mod.get_calc_batch, "open_infer_time",
                     "open_infer_batches", "open_infer_leaves"))
    return stats


def test_stage_timing(monkeypatch):
    device = get_device()
    sims = Config.train_simulation
    model = ValueCNN().to(device).eval()

    stats = _install_inference_timer(monkeypatch)

    t_opening = Timer(sync_cuda=True)
    t_mcts = Timer(sync_cuda=True)
    t_misc = Timer()
    t_post = Timer()
    t_augment = Timer(sync_cuda=True)

    # ---- 单局自对弈（带插桩，逻辑同 utils.generate_single_game）----
    with torch.no_grad():
        with t_opening:
            board = utils_mod.generate_random_board(model)

        mcts = MCTS(model)
        state_boards, state_policies = [], []
        root = None
        game_temp = random.uniform(Config.temp_min, Config.temp_max)

        for move_num in range(15 * 15):
            with t_mcts:
                (_v, probs), new_root = mcts.run(
                    board, sims, cur_root=root, return_root=True,
                    add_noise=True)

            with t_misc:
                state_boards.append(board_to_tensor(copy.deepcopy(board)))
                state_policies.append(torch.FloatTensor(probs))
                action = mcts_mod.calc_next_move(
                    board, probs, temperature=game_temp)
                board[action[0]][action[1]] = 1
                root = mcts.next_root(new_root, action)
                if mcts.is_terminal(board):
                    break
                flip_perspective(board)

        with t_post:
            n = len(state_boards)
            values = [0.0] * n
            weights = [1.0] * n
            boards_t = torch.stack(state_boards)
            policies_t = torch.stack(state_policies)

        with t_augment:
            aug = augment_data(boards_t, policies_t, values, weights)

    game_total = (t_opening.total + t_mcts.total + t_misc.total
                  + t_post.total + t_augment.total)

    # ---- 多进程管线开销 ----
    workers = min(Config.num_workers, os.cpu_count() or 4, 10)
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(processes=workers) as pool:
        # torch.Tensor 任务强制每个 worker 完成 spawn + import torch
        # （真实管线中 worker 还会 import src 各模块，耗时主要在 torch）
        pool.map(torch.Tensor, range(workers))
    pool_startup = time.perf_counter() - t0

    # worker 内每局都要构建模型并加载权重
    build_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        m = ValueCNN()
        m.load_state_dict(model.state_dict())
        m.to(device)
        build_times.append(time.perf_counter() - t0)
        del m
    model_build = sum(build_times) / len(build_times)

    # ---- 训练环节：单 batch forward+backward ----
    train_model = ValueCNN().to(device)
    optimizer = optim.Adam(train_model.parameters(), lr=1e-4)
    v_crit = nn.MSELoss(reduction="none")
    p_crit = nn.KLDivLoss(reduction="none")
    batch = Config.batch_size
    xb = torch.randn(batch, 3, 15, 15, device=device)
    yb = torch.zeros(batch, 1, device=device)
    pb = torch.rand(batch, 225, device=device)
    pb = pb / pb.sum(dim=1, keepdim=True)
    wb = torch.ones(batch, device=device)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    n_train_steps = 5
    t0 = time.perf_counter()
    for _ in range(n_train_steps):
        optimizer.zero_grad()
        pv, pp = train_model(xb)
        vl = v_crit(pv, yb).squeeze(1)
        pl = p_crit(F.log_softmax(pp, dim=1), pb).sum(dim=1)
        loss = 2 * (vl * wb).mean() + (pl * wb).mean()
        loss.backward()
        optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    per_batch = (time.perf_counter() - t0) / n_train_steps

    # ---- 汇总输出 ----
    def pct(t):
        return f"{t / game_total * 100:5.1f}%"

    infer_per_batch = (stats["mcts_infer_time"]
                       / max(stats["mcts_infer_batches"], 1) * 1000)
    leaves_per_batch = (stats["mcts_infer_leaves"]
                        / max(stats["mcts_infer_batches"], 1))
    tree_time = t_mcts.total - stats["mcts_infer_time"]
    aug_samples = len(aug[0])

    logger.info("=" * 72)
    logger.info("单局自对弈各环节耗时（批量推理 batch=%d / %d 模拟每步 / 设备 %s / 共 %d 手）",
                Config.infer_batch_size, sims, device, n)
    logger.info("-" * 72)
    logger.info("  %-26s %8s  %6s  %s", "环节", "耗时(s)", "占比", "备注")
    logger.info("  %-26s %8.2f  %6s  %d 批 / %d 个候选局面",
                "随机开局生成", t_opening.total, pct(t_opening.total),
                stats["open_infer_batches"], stats["open_infer_leaves"])
    logger.info("  %-26s %8.2f  %6s  %d 手 × %d 模拟",
                "MCTS 搜索合计", t_mcts.total, pct(t_mcts.total), n, sims)
    logger.info("  %-26s %8.2f  %6s  %d 批 / %d 叶子, 平均 %.1f ms/批(%.1f 叶/批)",
                "  ├ 网络推理 get_calc_batch", stats["mcts_infer_time"],
                pct(stats["mcts_infer_time"]), stats["mcts_infer_batches"],
                stats["mcts_infer_leaves"], infer_per_batch, leaves_per_batch)
    logger.info("  %-26s %8.2f  %6s  选择/扩展/回传(Python)",
                "  └ 树操作", tree_time, pct(tree_time))
    logger.info("  %-26s %8.2f  %6s",
                "落子/编码/树复用等", t_misc.total, pct(t_misc.total))
    logger.info("  %-26s %8.2f  %6s",
                "终局标签+tensor整理", t_post.total, pct(t_post.total))
    logger.info("  %-26s %8.2f  %6s  %d 局面 -> %d 条样本",
                "D4 数据增强(×8)", t_augment.total, pct(t_augment.total),
                n, aug_samples)
    logger.info("  %-26s %8.2f  %6s",
                "单局合计", game_total, "100.0%")
    logger.info("-" * 72)
    logger.info("多进程管线开销：")
    logger.info("  spawn 进程池启动+torch 导入(%d workers): %.2f 秒（每轮训练仅 1 次）",
                workers, pool_startup)
    logger.info("  worker 内模型构建+权重加载+上GPU: %.2f 秒/局", model_build)
    logger.info("训练环节：")
    logger.info("  单个 batch(%d) 训练步: %.3f 秒", batch, per_batch)
    logger.info("  外推每轮(100局≈3000样本, 3 epoch≈%d 步): %.1f 秒",
                int(3000 / batch) * 3, per_batch * int(3000 / batch) * 3)
    logger.info("=" * 72)

    assert game_total > 0
    # 叶子评估数：终局模拟走规则回传不评估，批量模式下同批模拟共享
    # 叶子会去重，因此叶子数显著少于模拟数（0.3 为防御性下界）
    assert stats["mcts_infer_leaves"] >= n * sims * 0.3
    # 批量确实生效：平均每批评估多于 1 个叶子
    assert stats["mcts_infer_leaves"] / max(stats["mcts_infer_batches"], 1) > 1.5
    # 样本数 = 实际记录的局面数（含最后终局手），增强后 ×8
    assert aug_samples == n * 8
