"""自对弈数据生成性能基准。

测量两部分：
1. 串行单局：单局真实计算耗时与平均步数；
2. 多进程 10 局：训练管线（generate_selfplay_data）的实际吞吐，
   报告总耗时、平均每局摊派耗时、相对串行的并行加速比。

注：本测试使用真实训练配置（train_simulation 次模拟/步），耗时较长，
属于性能基准而非功能测试。
"""

import logging
import os
import time

import pytest

from src.config import Config
from src.net import ValueCNN, get_device
from src.utils import generate_selfplay_data, generate_single_game

logger = logging.getLogger(__name__)

BENCH_GAMES = 10


@pytest.fixture(autouse=True)
def _enable_info_logging():
    """让基准统计的 INFO 日志在 -s 模式下可见。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    yield


def test_selfplay_10_games_timing(monkeypatch):
    """生成 10 局自对弈数据，统计平均每局耗时。"""
    workers = min(Config.num_workers, os.cpu_count() or 4, BENCH_GAMES)
    monkeypatch.setattr(Config, "num_workers", workers)
    monkeypatch.setattr(Config, "train_ratio", 0.9)

    model = ValueCNN().to(get_device()).eval()
    sims = Config.train_simulation

    # ---- 1. 串行单局基线 ----
    t0 = time.perf_counter()
    single = generate_single_game(
        0, model.state_dict(), sims,
        model.hidden_channels, model.num_blocks)
    serial_time = time.perf_counter() - t0
    single_steps = len(single[0])
    logger.info("【串行基线】1 局耗时 %.1f 秒，共 %d 步（%.2f 秒/步）",
                serial_time, single_steps, serial_time / max(single_steps, 1))

    # ---- 2. 多进程 10 局（训练管线真实路径）----
    t0 = time.perf_counter()
    train_data, val_data = generate_selfplay_data(
        model, num_games=BENCH_GAMES, num_simulations=sims)
    parallel_time = time.perf_counter() - t0

    total_samples = len(train_data[0]) + len(val_data[0])
    avg_per_game = parallel_time / BENCH_GAMES
    speedup = serial_time / avg_per_game

    logger.info("=" * 60)
    logger.info("自对弈性能基准：%d 局 / %d workers / %d 次模拟每步 / 设备 %s",
                BENCH_GAMES, workers, sims, get_device())
    logger.info("-" * 60)
    logger.info("10 局总耗时: %.1f 秒（%.2f 分钟）",
                parallel_time, parallel_time / 60)
    logger.info("平均每局耗时（并行摊派）: %.1f 秒", avg_per_game)
    logger.info("吞吐: %.2f 局/分钟", BENCH_GAMES / parallel_time * 60)
    logger.info("串行单局: %.1f 秒 -> 并行加速比: %.2fx（%d workers）",
                serial_time, speedup, workers)
    logger.info("增强后总样本 %d 条（D4 x8），平均每局原始局面约 %.0f 个",
                total_samples, total_samples / 8 / BENCH_GAMES)
    logger.info("=" * 60)

    # 基本正确性
    assert total_samples > 0
    # 性能护栏：每局摊派不超过 10 分钟（防止严重性能回退）
    assert avg_per_game < 600, f"每局耗时 {avg_per_game:.1f}s 超出护栏"
