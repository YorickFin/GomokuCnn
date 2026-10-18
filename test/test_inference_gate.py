"""计算侧推理优化的验收关卡（优化方案.md 第 2/5/10 节）。

阶段 2（channels_last / fp16 autocast）改变 forward 执行路径或数值，
按语义红线必须依次通过：

1. 基线逐位一致：统一推理路径 calc_batch_arrays 在默认开关下与旧
   实现（get_calc_batch）输出逐位相同（不改数值的硬保证）；
2. 固定评估集 loss：新旧推理路径在同一份固定评估集（models/run_3/
   eval_fixed.pt）上对比 value(MSE) / policy(KL)，fp16 路径不回退；
3. 新旧路径 arena：同一权重两条推理路径对弈 100 局，胜率显著偏离
   50% 即回退（binomial 2σ 带 0.40~0.60）。

运行：pytest test/test_inference_gate.py -s
"""

import numpy as np
import pytest
import torch

from src.config import Config, board_size
from src.net import ValueCNN, calc_batch_arrays, get_calc_batch, get_device, load_model

pytest.importorskip("torch")

MODEL_PATH = 'models/run_3/1400.pth'
EVAL_SET_PATH = 'models/run_3/eval_fixed.pt'


def _load_model():
    path = __import__('pathlib').Path(MODEL_PATH)
    if not path.is_file():
        pytest.skip(f"未找到模型 {MODEL_PATH}")
    device = get_device()
    model = load_model(str(path), device)
    return model, device


def _random_boards(n, seed=42):
    """n 个随机中盘 (n,225) int8（无五连）。"""
    from src import board as board_mod
    from src.board import flip_perspective
    rng = np.random.default_rng(seed)
    boards = []
    for _ in range(n):
        for _try in range(500):
            board = board_mod.new_board()
            player = 1
            for k in rng.choice(225, size=int(rng.integers(10, 36)),
                                replace=False):
                i, j = divmod(int(k), board_size)
                board[i][j] = player
                player = -player
            if player == -1:
                flip_perspective(board)
            if board_mod.evaluation_func(board) == 0:
                boards.append(np.asarray(board, dtype=np.int8).reshape(-1))
                break
        else:
            raise AssertionError("未找到无五连的随机局面")
    return np.stack(boards)


# ---------- 1. 基线逐位一致 ----------

def test_baseline_path_bitwise_vs_legacy():
    """calc_batch_arrays 默认开关 vs 旧 get_calc_batch 逐位一致。"""
    model, device = _load_model()
    boards = _random_boards(64)

    values_new, policies_new = calc_batch_arrays(model, boards, device)
    boards_2d = [b.reshape(board_size, board_size).tolist() for b in boards]
    results = get_calc_batch(model, boards_2d)
    values_old = np.array([v for v, _ in results])
    policies_old = np.array([p for _, p in results]).reshape(len(boards), -1)

    np.testing.assert_array_equal(values_new, values_old)
    np.testing.assert_array_equal(policies_new, policies_old)


def test_channels_last_path_same_math():
    """channels_last 与基线的输出差异应在 fp32 舍入量级（同一数学）。"""
    model, device = _load_model()
    boards = _random_boards(64)
    model_cl = model.to(memory_format=torch.channels_last)

    values_base, policies_base = calc_batch_arrays(model, boards, device)
    values_cl, policies_cl = calc_batch_arrays(
        model_cl, boards, device, channels_last=True)
    model.to(memory_format=torch.contiguous_format)

    np.testing.assert_allclose(values_cl, values_base, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(policies_cl, policies_base, rtol=1e-5, atol=1e-6)


# ---------- 2. 固定评估集 loss ----------

def _eval_loss(model, device, data, **kwargs):
    """按统一推理路径计算固定评估集 (value MSE, policy KL) 平均损失。

    与 train.evaluate_on_fixed 同口径（MSE + KLDivLoss batchmean），
    但 forward 走 calc_batch_arrays（受 channels_last/fp16 开关控制）。
    """
    from pathlib import Path
    if not Path(EVAL_SET_PATH).is_file():
        pytest.skip(f"未找到固定评估集 {EVAL_SET_PATH}")

    boards_t, policies_t, values_t, _weights = data
    n = boards_t.shape[0]
    # (3,15,15) float 编码还原 int8 棋盘：ch0=当前方, ch1=对手
    board_int = (boards_t[:, 0] - boards_t[:, 1]).round().to(torch.int8)
    board_flat = board_int.reshape(n, -1).numpy()

    total_value = 0.0
    total_policy = 0.0
    bs = Config.batch_size
    for lo in range(0, n, bs):
        hi = min(lo + bs, n)
        values, policies = calc_batch_arrays(
            model, board_flat[lo:hi], device, **kwargs)
        v = torch.tensor(values, dtype=torch.float32)
        p = torch.tensor(policies, dtype=torch.float32)
        target_v = values_t[lo:hi]
        target_p = policies_t[lo:hi].reshape(hi - lo, -1)
        total_value += torch.nn.functional.mse_loss(v, target_v).item()
        log_pred = torch.log(p.clamp_min(1e-12))
        total_policy += torch.nn.functional.kl_div(
            log_pred, target_p, reduction='batchmean').item()

    n_batches = (n + bs - 1) // bs
    return total_value / n_batches, total_policy / n_batches


@pytest.fixture(scope='module')
def eval_data():
    from pathlib import Path
    if not Path(EVAL_SET_PATH).is_file():
        pytest.skip(f"未找到固定评估集 {EVAL_SET_PATH}")
    from src.utils import load_fixed_eval_set
    return load_fixed_eval_set(EVAL_SET_PATH)


def test_fixed_eval_loss_variants(eval_data):
    """各推理变体在固定评估集上的 loss 对比（fp16 不回退判据）。

    输出对比表；判据：fp16 变体 total loss 相对基线变化 < 5%
    （fp16 量化噪声不应对标签拟合产生系统性劣化）。
    """
    model, device = _load_model()
    variants = [
        ('基线', dict()),
        ('channels_last', dict(channels_last=True)),
        ('fp16', dict(fp16=True)),
        ('channels_last+fp16', dict(channels_last=True, fp16=True)),
    ]
    results = {}
    print("\n[固定评估集 loss 对比]")
    for name, kwargs in variants:
        if kwargs.get('channels_last'):
            model.to(memory_format=torch.channels_last)
        v_loss, p_loss = _eval_loss(model, device, eval_data, **kwargs)
        results[name] = (v_loss, p_loss)
        print(f"  {name:20s}: value={v_loss:.6f}  policy={p_loss:.6f}  "
              f"total={2 * v_loss + p_loss:.6f}")
        if kwargs.get('channels_last'):
            model.to(memory_format=torch.contiguous_format)

    base_total = 2 * results['基线'][0] + results['基线'][1]
    fp16_total = 2 * results['channels_last+fp16'][0] \
        + results['channels_last+fp16'][1]
    delta = (fp16_total - base_total) / base_total
    print(f"  组合变体 total 相对基线变化: {delta:+.2%}")
    assert abs(delta) < 0.05, \
        f"fp16 组合变体 loss 回退 {delta:+.2%}（阈值 5%），不应启用"


# ---------- 3. 新旧路径 arena ----------

def test_arena_inference_variants():
    """同一权重、新旧推理路径 arena 100 局（预期 ≈50%）。

    a = channels_last + fp16（候选路径），b = 基线路径。
    胜率带 0.40~0.60（100 局 binomial 2σ≈0.10），显著偏离即回退。
    """
    if not torch.cuda.is_available():
        pytest.skip("无 CUDA 设备")
    model, _device = _load_model()

    from src.eval import pit
    result = pit(
        model, model, num_games=100, simulations=Config.arena_simulations,
        opening_moves=Config.arena_opening_moves,
        mcts_kwargs_a=dict(channels_last=True, fp16=True),
        mcts_kwargs_b=dict())
    wr = result['win_rate_a']
    print(f"\n[推理路径 arena] 候选(channels_last+fp16) vs 基线 "
          f"{result['wins_a']}:{result['wins_b']} "
          f"(平 {result['draws']})，胜率 {wr:.2%}")
    assert 0.40 <= wr <= 0.60, \
        f"新路径胜率 {wr:.2%} 显著偏离 50%，应回退（方案 10 节风险清单）"


if __name__ == '__main__':
    test_baseline_path_bitwise_vs_legacy()
    test_channels_last_path_same_math()
    test_arena_inference_variants()
