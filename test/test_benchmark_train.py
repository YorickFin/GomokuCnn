"""训练阶段性能基准：网络结构固定（Config.channel/num_blocks），
测训练管线各环节耗时与优化开关收益。

不是断言型功能测试，属于性能基准，运行约 20~30 分钟：
    .venv\\Scripts\\python.exe -m pytest test/test_benchmark_train.py -s

流程：
0. 数据准备：benchmark/bench_games_100.pt 不存在时，用
   models/run_4/champion.pth 生成 100 局自对弈（训练口径：30 sims、
   全节点收集、per-game 随机温度）并存盘；此后所有配置共用同一份数据。
1. 配置矩阵（每配置模拟完整一轮训练 = 3 epochs，结构不变）：
   C1 baseline        ：现状复刻（逐样本 D4 物化 + DataLoader 每 epoch
                         shuffle/collate + fp32）
   C2 aug_vec         ：向量化 D4 增强（整批 rot90/flip，一次 8 倍）
   C3 direct_batch    ：C2 + randperm 直切 batch（绕开 DataLoader/collate，
                         每 epoch 重新 shuffle）
   C4 io_opt          ：C3 + 非阻塞 H2D + cudnn.benchmark + channels_last
   C5 amp             ：C4 + bf16 autocast（Ampere 及以上）
   C6/C7              ：C5 + batch 1024 / 2048
2. 分段计时：增强物化 / 取批(collate或切片) / H2D / GPU前向反向。
3. 输出对比表（含末 epoch loss，确认各数值路径无崩坏）。

注意：
- 每配置重建模型与 optimizer，固定随机种子保证 batch 顺序一致；
- 第 1 个 epoch 含 cudnn autotune 预热，表格耗时取第 2/3 epoch 均值；
- AMP 用 bf16（与 fp32 同 dynamic range，无需 GradScaler）。
"""

import copy
import logging
import multiprocessing
import os
import statistics
import sys
import time

import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.net import ValueCNN, get_device
from src.utils import (_generate_games, augment_data, concat_games,
                       split_games)

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(ROOT, "benchmark")
BENCH_DATA = os.path.join(BENCH_DIR, "bench_games_100.pt")
BASE_MODEL = os.path.join(ROOT, "models", "run_4", "champion.pth")

EPOCHS = 3                    # 对齐真实一轮训练的 epoch 数
TIME_EPOCHS = (2, 3)          # 第 2/3 epoch 计时（跳过 autotune 预热）


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


# ---------------------------------------------------------------- 数据准备

def ensure_bench_data():
    """确保固定基准数据存在；不存在时用 run_4 champion 生成 100 局。"""
    if os.path.exists(BENCH_DATA):
        return BENCH_DATA

    assert os.path.exists(BASE_MODEL), f"找不到基础模型: {BASE_MODEL}"
    os.makedirs(BENCH_DIR, exist_ok=True)

    device = get_device()
    state = torch.load(BASE_MODEL, map_location=device, weights_only=True)
    model = ValueCNN.from_state_dict(state).to(device)
    model.eval()
    logger.info("基础模型: %s（%dch/%dblk）", BASE_MODEL,
                model.hidden_channels, model.num_blocks)

    t0 = time.perf_counter()
    with multiprocessing.get_context('spawn').Pool(
            processes=Config.num_workers) as pool:
        games = _generate_games(model, 100,
                                num_simulations=Config.train_simulation,
                                pool=pool)
    dt = time.perf_counter() - t0
    logger.info("已生成 100 局基准数据（%.1f 秒，共 %d 原始样本），保存至 %s",
                dt, sum(len(g[2]) for g in games), BENCH_DATA)
    torch.save(games, BENCH_DATA)
    return BENCH_DATA


# ------------------------------------------------- 向量化 D4 增强（C2 起）

def vec_augment_data(boards, policies, values, weights):
    """整批 D4 增强：8 个视图各用一次 rot90/flip，替代逐样本循环。

    输出与 augment_data 逐位一致：k 外层、flip 内层（k0,k0f,k1,k1f...）
    决定视图顺序，stack(dim=1).reshape 使每样本 8 视图连续（sample-major），
    values/weights 用 repeat_interleave 对齐同一排布。
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


def test_vec_augment_parity():
    """功能校验：vec_augment_data 与 augment_data 输出逐位一致。"""
    torch.manual_seed(0)
    n = 5
    boards = torch.randn(n, 3, 15, 15)
    policies = torch.rand(n, 15, 15)
    values = torch.randn(n)
    weights = torch.rand(n)
    ref = augment_data(boards, policies, values, weights)
    out = vec_augment_data(boards, policies, values, weights)
    for a, b, label in zip(ref, out,
                           ("boards", "policies", "values", "weights")):
        assert torch.equal(a, b), f"{label} 不一致"
        assert a.shape == b.shape


# ------------------------------------------------------------ 训练基准

def _iter_loader(data, batch_size, shuffle):
    """baseline 取批：DataLoader 默认 collate，每 epoch 重新 shuffle。"""
    from torch.utils.data import DataLoader
    from src.utils import WeightedDataset
    ds = WeightedDataset(*data, augment=False)
    yield from DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _iter_slices(data, batch_size, shuffle):
    """直切取批：randperm 后大 tensor 高级索引切片（一次 copy，零 collate）。"""
    boards, policies, values, weights = data
    n = boards.shape[0]
    perm = torch.randperm(n) if shuffle else torch.arange(n)
    for s in range(0, n, batch_size):
        idx = perm[s:s + batch_size]
        yield (boards[idx], policies[idx], values[idx], weights[idx])


def _run_epoch(model, device, iter_fn, optimizer, t_batch, t_h2d, t_gpu,
               train_mode):
    """遍历一个 epoch，返回 (value_loss, policy_loss)。

    iter_fn 每次调用返回新的 batch 迭代器（DataLoader 每 epoch 重新
    shuffle/collate；直切每 epoch 重新 randperm），取批耗时计 t_batch。
    amp 为 None 时走 fp32，否则为 dtype（bfloat16）。
    """
    value_criterion = nn.MSELoss(reduction='none')
    policy_criterion = nn.KLDivLoss(reduction='none')
    amp = getattr(model, "_bench_amp", None)
    n = 0
    tv = tp = 0.0
    for boards, policies, values, weights in iter_fn():
        with t_batch:
            pass
        with t_h2d:
            boards = boards.to(device, non_blocking=True)
            policies = policies.to(device, non_blocking=True).view(
                policies.size(0), -1)
            values = values.to(device, non_blocking=True).unsqueeze(1)
            weights = weights.to(device, non_blocking=True)

        if train_mode:
            with t_gpu:
                optimizer.zero_grad()
                if amp is not None:
                    with torch.autocast('cuda', dtype=amp):
                        pred_values, pred_policies = model(boards)
                    pred_values, pred_policies = (
                        pred_values.float(), pred_policies.float())
                else:
                    pred_values, pred_policies = model(boards)
                value_loss = value_criterion(
                    pred_values, values).squeeze(1)
                policy_loss = policy_criterion(
                    F.log_softmax(pred_policies, dim=1), policies).sum(dim=1)
                loss = (2 * (value_loss * weights).mean()
                        + (policy_loss * weights).mean())
                loss.backward()
                optimizer.step()
            tv += (value_loss * weights).mean().item()
            tp += (policy_loss * weights).mean().item()
        else:
            with t_gpu:
                if amp is not None:
                    with torch.autocast('cuda', dtype=amp):
                        pred_values, pred_policies = model(boards)
                    pred_values, pred_policies = (
                        pred_values.float(), pred_policies.float())
                else:
                    pred_values, pred_policies = model(boards)
            tv += nn.functional.mse_loss(pred_values, values).item()
            tp += nn.functional.kl_div(
                F.log_softmax(pred_policies, dim=1),
                policies, reduction='batchmean').item()
            n += 1
    return (tv / max(n, 1), tp / max(n, 1)) if not train_mode else (tv, tp)


def bench_once(name, games, *, vec_aug, direct_batch, io_opt, amp,
               batch_size):
    """按开关组合测一轮完整训练（3 epochs），返回分段耗时与 loss。"""
    device = get_device()
    torch.manual_seed(42)

    torch.backends.cudnn.benchmark = bool(io_opt)

    t_aug = Timer(sync_cuda=False)
    t_batch = Timer(sync_cuda=False)   # 取批（collate / 切片）
    t_h2d = Timer(sync_cuda=True)
    t_gpu = Timer(sync_cuda=True)

    # ---- 数据准备（模拟 train.py 每轮流程）----
    train_games, val_games = split_games(copy.deepcopy(games),
                                         Config.train_ratio)
    val_raw = concat_games(val_games)
    with t_aug:
        val_data = (vec_augment_data(*val_raw) if vec_aug
                    else augment_data(*val_raw))
    train_raw = concat_games(train_games)
    with t_aug:
        train_data = (vec_augment_data(*train_raw) if vec_aug
                      else augment_data(*train_raw))

    if io_opt:
        # channels_last 物化（模拟把输入预转 NHWC；实际训练可在 batch 供给层做）
        train_data = (train_data[0].contiguous(
            memory_format=torch.channels_last), *train_data[1:])
        val_data = (val_data[0].contiguous(
            memory_format=torch.channels_last), *val_data[1:])

    # ---- 模型 / 优化器 ----
    model = ValueCNN().to(device)
    if io_opt:
        model = model.to(memory_format=torch.channels_last)
    model._bench_amp = (torch.bfloat16 if amp else None)
    # plain Adam 无 weight_decay，对齐 train_model 的 nn_012 LR 方案
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    if direct_batch:
        train_iter = lambda: _iter_slices(train_data, batch_size, True)
    else:
        train_iter = lambda: _iter_loader(train_data, batch_size, True)
    if direct_batch:
        val_iter = lambda: _iter_slices(val_data, batch_size, False)
    else:
        val_iter = lambda: _iter_loader(val_data, batch_size, False)

    epoch_times = []
    last_train = last_val = (0.0, 0.0)
    for ep in range(1, EPOCHS + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tv, tp = _run_epoch(model, device, train_iter, optimizer,
                            t_batch, t_h2d, t_gpu, train_mode=True)
        model.eval()
        vv, vp = _run_epoch(model, device, val_iter, optimizer,
                            t_batch, t_h2d, t_gpu, train_mode=False)
        model.train()
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        last_train = (tv, tp)
        last_val = (vv, vp)
        logger.info("  [%s] epoch %d/%d: %.1fs (train v=%.4f p=%.4f | "
                    "val v=%.4f p=%.4f)", name, ep, EPOCHS, dt, tv, tp, vv, vp)

    timed = [epoch_times[e - 1] for e in TIME_EPOCHS]
    result = {
        "name": name,
        "aug": t_aug.total,
        "batch": t_batch.total,
        "h2d": t_h2d.total,
        "gpu": t_gpu.total,
        "epoch_mean": statistics.mean(timed),
        "round3": sum(epoch_times),
        "train_loss": last_train,
        "val_loss": last_val,
        "samples": len(train_data[0]),
    }
    model._bench_amp = None
    torch.backends.cudnn.benchmark = False
    return result


CONFIGS = [
    ("C1 baseline", dict(vec_aug=False, direct_batch=False, io_opt=False,
                         amp=False, batch_size=512)),
    ("C2 aug_vec", dict(vec_aug=True, direct_batch=False, io_opt=False,
                        amp=False, batch_size=512)),
    ("C3 +direct_batch", dict(vec_aug=True, direct_batch=True, io_opt=False,
                              amp=False, batch_size=512)),
    ("C4 +io_opt", dict(vec_aug=True, direct_batch=True, io_opt=True,
                        amp=False, batch_size=512)),
    ("C5 +amp_bf16", dict(vec_aug=True, direct_batch=True, io_opt=True,
                          amp=True, batch_size=512)),
    ("C6 batch=1024", dict(vec_aug=True, direct_batch=True, io_opt=True,
                           amp=True, batch_size=1024)),
    ("C7 batch=2048", dict(vec_aug=True, direct_batch=True, io_opt=True,
                           amp=True, batch_size=2048)),
    ("C8 bf16_nchw", dict(vec_aug=True, direct_batch=True, io_opt=False,
                          amp=True, batch_size=2048)),
    ("C9 batch=4096", dict(vec_aug=True, direct_batch=True, io_opt=False,
                           amp=True, batch_size=4096)),
]


def run_benchmark(only=None):
    """only: 配置名列表（支持子串匹配），None 时跑全部。"""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ensure_bench_data()
    games = torch.load(BENCH_DATA, map_location='cpu', weights_only=True)
    logger.info("基准数据: %d 局，共 %d 原始样本", len(games),
                sum(len(g[2]) for g in games))

    selected = [(n, kw) for n, kw in CONFIGS
                if only is None or any(o in n for o in only)]
    results = [bench_once(n, games, **kw) for n, kw in selected]

    results_path = os.path.join(BENCH_DIR, "results.jsonl")
    for r in results:
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(repr(r) + "\n")
        print(f"[done] {r['name']}: 3epochs={r['round3']:.1f}s "
              f"(增强 {r['aug']:.1f}s, 取批 {r['batch']:.1f}s, "
              f"H2D {r['h2d']:.1f}s, GPU {r['gpu']:.1f}s)", flush=True)

    base = results[0]["round3"]
    lines = ["", "=" * 104]
    lines.append("%-18s %8s %8s %8s %8s %10s %10s %8s  %s" % (
        "配置", "增强s", "取批s", "H2Ds", "GPUs", "epoch均值s",
        "3epochs", "加速比", "val(v/p)"))
    for r in results:
        lines.append(
            "%-18s %8.2f %8.2f %8.2f %8.1f %10.1f %10.1f %7.2fx  "
            "%.4f/%.4f  (训练样本 %d)" % (
                r["name"], r["aug"], r["batch"], r["h2d"], r["gpu"],
                r["epoch_mean"], r["round3"], base / r["round3"],
                r["val_loss"][0], r["val_loss"][1], r["samples"]))
    lines.append("=" * 104)
    lines.append("说明: 增强s 为物化 8 倍数据 1 次（train.py 每轮 1 次）；"
                 "取批s/H2Ds/GPUs 为 3 epochs 累计；epoch均值s 为第 %d/%d "
                 "epoch 均值；加速比基于 3 epochs 总耗时" % TIME_EPOCHS)
    table = "\n".join(lines)
    print(table, flush=True)
    logger.info("\n%s", table)
    return results


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="训练基准需要 CUDA")
def test_benchmark_train():
    run_benchmark()


if __name__ == "__main__":
    run_benchmark(only=sys.argv[1:] or None)
