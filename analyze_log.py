# -*- coding: utf-8 -*-
"""五子棋 CNN 训练日志分析脚本（适配新版 train.py 日志格式）。

新版日志特征：
- 每轮依次输出：===== 训练轮次 N ===== → 自对弈 MCTS 模拟次数: S →
  回放池: X/Y 轮，共 Z 局原始数据（取样时在线 D4 增强） → 各 epoch 指标 →
  train/val losses → [固定评估]（→ 每 arena_interval 轮的保存与 [竞技场]）。
- 自对弈阶段以"回放池"行为结束标志（该行在自对弈完成后、训练开始前输出）；
  训练阶段以 val losses 行为结束标志。
- 回放池只存原始局（训练时统一 D4 增强），日志中只有局数，无每轮样本数。
- 模拟次数支持线性爬升：配置行为
  "配置: 自对弈 N 局/轮, 模拟次数 A→B/步（前 R 轮线性爬升，之后保持上限）, ..."。

用法:
    uv run python analyze_log.py [日志文件路径]

不传路径时，默认分析脚本同级 logging/ 目录下最新的 .log 文件。
输出目录为日志旁的 <日志名>_分析/，内含:
    - 分析报告.txt
    - 每轮指标.csv
    - 损失曲线.png / 学习率与数据量.png / 耗时分析.png / 竞技场胜率.png
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Windows 控制台中文输出兜底
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------- 日志解析

HEADER_RE = re.compile(
    r"^(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})"
    r" - (\w+) - ([\w.]+) - (.*)$"
)

RE_DEVICE = re.compile(r"开始训练，使用设备: (\w+)")
RE_RUN_DIR = re.compile(r"^输出目录: (.+)$")
RE_CONFIG_RAMP = re.compile(
    r"配置: 自对弈 (\d+) 局/轮, 模拟次数 (\d+)→(\d+)/步"
    r"（前 (\d+) 轮线性爬升，之后保持上限）, batch=(\d+), epoch=(\d+), 回放 (\d+) 轮"
)
RE_CONFIG_FIXED = re.compile(
    r"配置: 自对弈 (\d+) 局/轮, 模拟 (\d+) 次/步, batch=(\d+), "
    r"epoch=(\d+), 回放 (\d+) 轮"
)
RE_BASE_USER = re.compile(r"^已加载基础模型（用户指定）: (.+)$")
RE_BASE_PREV = re.compile(r"^已加载基础模型（上一 run champion）: (.+)$")
RE_BASE_RANDOM = re.compile(r"无历史 champion，使用全新随机初始化模型")
RE_WORKERS = re.compile(r"持久工作进程池: 已创建 (\d+) 个 worker")
RE_EVALSET_GEN = re.compile(
    r"固定评估集: 未找到缓存 .+，用基础模型生成 (\d+) 局（不参与训练）")
RE_EVALSET_SAVED = re.compile(
    r"固定评估集: (?:已生成并保存至|已加载缓存) .+（增强后 (\d+) 样本）")
RE_ROUND = re.compile(r"===== 训练轮次 (\d+) =====")
RE_SIMS = re.compile(r"自对弈 MCTS 模拟次数: (\d+)")
RE_REPLAY = re.compile(
    r"回放池: (\d+)/(\d+) 轮，共 (\d+) 局原始数据（(?:训练时|取样时在线|全量) D4 .*增强）")
RE_EPOCH = re.compile(
    r"\[epoch (\d+)\] train value=([\d.]+) policy=([\d.]+) \| "
    r"val value=([\d.]+) policy=([\d.]+) \| lr=([\d.eE+-]+)"
)
RE_TRAIN_LOSSES = re.compile(r"train losses: \[([^\]]*)\]")
RE_VAL_LOSSES = re.compile(r"val losses: \[([^\]]*)\]")
RE_SAVED = re.compile(r"^模型已保存为 (.+)$")
RE_FIXED_EVAL = re.compile(
    r"\[固定评估\] value=([\d.]+) policy=([\d.]+) \| 总损失=([\d.]+)")
RE_LR_STATUS = re.compile(
    r"\[LR\] lr=([\d.eE+-]+) best=([\d.]+) bad_epochs=(\d+)/(\d+)")
RE_LR_COSINE = re.compile(
    r"\[LR\] lr=([\d.eE+-]+)（余弦相位 (\d+)/(\d+)）")
RE_LR_VALLEY = re.compile(
    r"\[LR\] lr=([\d.eE+-]+)（谷底停留第 (\d+) 轮，arena 连续无晋升 (\d+)/(\d+)）")
RE_LR_JUMP = re.compile(r"\[LR\] 谷底结束：.*自 r(\d+) 起")
RE_ARENA = re.compile(
    r"\[竞技场\] 轮次 (\d+) 对 champion\(轮次(-?\d+)\)："
    r"(\d+) 局 胜 (\d+) 负 (\d+) 平 (\d+)，胜率 ([\d.]+)%")
RE_ARENA_PROMOTED = re.compile(r"\[竞技场\] 胜率达到 ([\d.]+)% 阈值")
RE_BEST_SAVED = re.compile(r"\[best\] val 滑动平均创新低 ([\d.]+)（轮次 (\d+)）")
RE_ARENA_KEPT = re.compile(r"\[竞技场\] 胜率未达 ([\d.]+)% 阈值")
RE_TRAIN_DONE = re.compile(r"训练全部完成（轮次 (\d+) ~ (\d+)）")


@dataclass
class EpochStat:
    epoch: int
    train_value: float
    train_policy: float
    val_value: float
    val_policy: float
    lr: float


@dataclass
class ArenaStat:
    step: int
    champion_step: int
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float          # 0~1
    promoted: bool = False


@dataclass
class RoundStat:
    step: int
    t_round: datetime
    num_sims: int | None = None          # 本轮自对弈 MCTS 模拟次数
    replay_filled: int = 0               # 回放池已填充轮数
    replay_cap: int = 0                  # 回放池容量（轮）
    replay_games: int = 0                # 回放池中原始局数（未增强）
    t_selfplay_end: datetime | None = None   # "回放池"行时刻 = 自对弈结束
    t_trained: datetime | None = None        # val losses 行时刻 = 训练结束
    t_saved: datetime | None = None          # checkpoint 保存时刻
    t_last_event: datetime | None = None     # 本轮最后一条日志时刻（含固定评估/竞技场）
    epochs: list[EpochStat] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    model_path: str = ""
    # 固定评估集（跨轮次可比）
    fixed_value: float | None = None
    fixed_policy: float | None = None
    fixed_total: float | None = None
    # LR 调度状态（每轮的 [LR] 行）：状态机（新）或 plateau（旧 run_6/7/8）
    lr: float | None = None
    lr_phase: int | None = None       # 余弦下降段相位（0 ~ 周期-1）
    lr_phase_len: int | None = None   # 下降段长度（轮）
    lr_valley: int | None = None      # 谷底停留第 N 轮（1 起）
    lr_fail: int | None = None        # 谷底 arena 连续无晋升计数
    lr_fail_max: int | None = None    # 跳回阈值
    lr_jump_round: int | None = None  # 跳回事件（新周期起始轮）
    lr_best: float | None = None      # 以下仅 plateau 旧日志
    lr_bad: int | None = None
    lr_patience: int | None = None

    @property
    def checkpoint(self) -> int:
        return self.step + 1

    @property
    def completed(self) -> bool:
        # 训练是否跑完：以 val losses 日志（训练结束）为准，
        # 与是否保存 checkpoint 解耦——checkpoint 每 arena_interval 轮才存一次。
        return self.t_trained is not None and bool(self.epochs)

    @property
    def has_checkpoint(self) -> bool:
        return self.t_saved is not None and bool(self.model_path)

    @property
    def selfplay_sec(self) -> float | None:
        if self.t_selfplay_end is None:
            return None
        return (self.t_selfplay_end - self.t_round).total_seconds()

    @property
    def train_sec(self) -> float | None:
        if self.t_selfplay_end is None or self.t_trained is None:
            return None
        return (self.t_trained - self.t_selfplay_end).total_seconds()

    @property
    def total_sec(self) -> float | None:
        end = self.t_last_event or self.t_trained or self.t_saved
        if end is None:
            return None
        return (end - self.t_round).total_seconds()

    @property
    def last_epoch(self) -> EpochStat | None:
        return self.epochs[-1] if self.epochs else None


@dataclass
class LogData:
    path: Path
    device: str = ""
    run_dir: str = ""
    config: dict = field(default_factory=dict)
    base_model: str = ""                 # 基础模型来源描述
    workers: int | None = None
    evalset_games: int | None = None     # 固定评估集局数
    evalset_samples: int | None = None   # 固定评估集增强后样本数
    rounds: list[RoundStat] = field(default_factory=list)
    arenas: list[ArenaStat] = field(default_factory=list)
    arena_promote_rate: float | None = None   # 百分数（如 50.0），来自竞技场日志
    best_smooth: float | None = None     # best.pth 判据（val 滑动平均最优）
    best_step: int | None = None         # best.pth 最后保存的轮次
    train_done_range: tuple[int, int] | None = None  # "训练全部完成"的轮次范围
    t_first: datetime | None = None
    t_last: datetime | None = None
    parse_warnings: list[str] = field(default_factory=list)


def _parse_ts(m: re.Match) -> datetime:
    return datetime(
        int(m.group(1)), int(m.group(2)), int(m.group(3)),
        int(m.group(4)), int(m.group(5)), int(m.group(6)),
    )


def _parse_loss_list(text: str) -> list[float]:
    return [float(x.strip().strip("'\"")) for x in text.split(",") if x.strip()]


def parse_log(path: Path) -> LogData:
    """逐行解析日志，返回结构化 LogData。"""
    data = LogData(path=path)
    current: RoundStat | None = None

    def _touch(ts: datetime) -> None:
        if current is not None:
            current.t_last_event = ts

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            hm = HEADER_RE.match(line)
            if not hm:
                if line.strip():
                    data.parse_warnings.append(f"第 {lineno} 行无法识别: {line[:80]}")
                continue
            ts = _parse_ts(hm)
            msg = hm.group(9)
            if data.t_first is None:
                data.t_first = ts
            data.t_last = ts

            if m := RE_DEVICE.search(msg):
                data.device = m.group(1)
            elif m := RE_RUN_DIR.search(msg):
                data.run_dir = m.group(1).strip()
            elif m := RE_CONFIG_RAMP.search(msg):
                data.config = {
                    "自对弈局数/轮": int(m.group(1)),
                    "模拟次数爬升": f"{m.group(2)}→{m.group(3)}"
                                   f"（前 {m.group(4)} 轮线性爬升）",
                    "batch": int(m.group(5)),
                    "epoch": int(m.group(6)),
                    "回放轮数": int(m.group(7)),
                }
            elif m := RE_CONFIG_FIXED.search(msg):
                data.config = {
                    "自对弈局数/轮": int(m.group(1)),
                    "模拟次数/步": int(m.group(2)),
                    "batch": int(m.group(3)),
                    "epoch": int(m.group(4)),
                    "回放轮数": int(m.group(5)),
                }
            elif m := RE_BASE_USER.search(msg):
                data.base_model = f"用户指定: {m.group(1).strip()}"
            elif m := RE_BASE_PREV.search(msg):
                data.base_model = f"上一 run champion: {m.group(1).strip()}"
            elif RE_BASE_RANDOM.search(msg):
                data.base_model = "全新随机初始化"
            elif m := RE_WORKERS.search(msg):
                data.workers = int(m.group(1))
            elif m := RE_EVALSET_GEN.search(msg):
                data.evalset_games = int(m.group(1))
            elif m := RE_EVALSET_SAVED.search(msg):
                data.evalset_samples = int(m.group(1))
            elif m := RE_ROUND.search(msg):
                current = RoundStat(step=int(m.group(1)), t_round=ts)
                data.rounds.append(current)
            elif current is not None and (m := RE_SIMS.search(msg)):
                current.num_sims = int(m.group(1))
                _touch(ts)
            elif current is not None and (m := RE_REPLAY.search(msg)):
                current.replay_filled = int(m.group(1))
                current.replay_cap = int(m.group(2))
                current.replay_games = int(m.group(3))
                current.t_selfplay_end = ts   # 自对弈结束 / 训练开始
                _touch(ts)
            elif current is not None and (m := RE_EPOCH.search(msg)):
                current.epochs.append(EpochStat(
                    epoch=int(m.group(1)),
                    train_value=float(m.group(2)),
                    train_policy=float(m.group(3)),
                    val_value=float(m.group(4)),
                    val_policy=float(m.group(5)),
                    lr=float(m.group(6)),
                ))
                _touch(ts)
            elif current is not None and (m := RE_TRAIN_LOSSES.search(msg)):
                current.train_losses = _parse_loss_list(m.group(1))
                _touch(ts)
            elif current is not None and (m := RE_VAL_LOSSES.search(msg)):
                current.val_losses = _parse_loss_list(m.group(1))
                current.t_trained = ts
                _touch(ts)
            elif current is not None and (m := RE_SAVED.search(msg)):
                current.t_saved = ts
                current.model_path = m.group(1).strip()
                _touch(ts)
            elif current is not None and (m := RE_FIXED_EVAL.search(msg)):
                current.fixed_value = float(m.group(1))
                current.fixed_policy = float(m.group(2))
                current.fixed_total = float(m.group(3))
                _touch(ts)
            elif current is not None and (m := RE_LR_VALLEY.search(msg)):
                current.lr = float(m.group(1))
                current.lr_valley = int(m.group(2))
                current.lr_fail = int(m.group(3))
                current.lr_fail_max = int(m.group(4))
                _touch(ts)
            elif current is not None and (m := RE_LR_COSINE.search(msg)):
                current.lr = float(m.group(1))
                current.lr_phase = int(m.group(2))
                current.lr_phase_len = int(m.group(3))
                _touch(ts)
            elif current is not None and (m := RE_LR_JUMP.search(msg)):
                current.lr_jump_round = int(m.group(1))
                _touch(ts)
            elif current is not None and (m := RE_LR_STATUS.search(msg)):
                current.lr = float(m.group(1))
                current.lr_best = float(m.group(2))
                current.lr_bad = int(m.group(3))
                current.lr_patience = int(m.group(4))
                _touch(ts)
            elif current is not None and (m := RE_BEST_SAVED.search(msg)):
                data.best_smooth = float(m.group(1))
                data.best_step = int(m.group(2))
                _touch(ts)
            elif m := RE_ARENA.search(msg):
                data.arenas.append(ArenaStat(
                    step=int(m.group(1)),
                    champion_step=int(m.group(2)),
                    games=int(m.group(3)),
                    wins=int(m.group(4)),
                    losses=int(m.group(5)),
                    draws=int(m.group(6)),
                    win_rate=float(m.group(7)) / 100.0,
                ))
                if data.rounds and data.rounds[-1].step == int(m.group(1)):
                    data.rounds[-1].t_last_event = ts
            elif m := RE_ARENA_PROMOTED.search(msg):
                data.arena_promote_rate = float(m.group(1))
                if data.arenas:
                    data.arenas[-1].promoted = True
                if data.rounds and data.rounds[-1].step == data.arenas[-1].step:
                    data.rounds[-1].t_last_event = ts
            elif m := RE_ARENA_KEPT.search(msg):
                if data.arena_promote_rate is None:
                    data.arena_promote_rate = float(m.group(1))
                if data.rounds and data.rounds[-1].step == data.arenas[-1].step:
                    data.rounds[-1].t_last_event = ts
            elif m := RE_TRAIN_DONE.search(msg):
                data.train_done_range = (int(m.group(1)), int(m.group(2)))

    return data


# ---------------------------------------------------------------- 报告生成

def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}小时{m:02d}分{s:02d}秒"
    if m:
        return f"{m}分{s:02d}秒"
    return f"{s}秒"


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _min_round(rounds: list[RoundStat], key) -> tuple[RoundStat, float]:
    best = min(rounds, key=key)
    return best, key(best)


def build_report(data: LogData) -> str:
    rounds = [r for r in data.rounds if r.completed]
    incomplete = [r for r in data.rounds if not r.completed]
    lines: list[str] = []

    add = lines.append
    add("=" * 64)
    add("五子棋 CNN 训练日志分析报告")
    add("=" * 64)
    add(f"日志文件: {data.path.name}")
    if data.run_dir:
        add(f"输出目录: {data.run_dir}")
    if data.t_first and data.t_last:
        span = (data.t_last - data.t_first).total_seconds()
        add(f"时间跨度: {data.t_first:%Y-%m-%d %H:%M:%S} ~ "
            f"{data.t_last:%Y-%m-%d %H:%M:%S}（总时长 {_fmt_dur(span)}）")
    if data.device:
        add(f"设备: {data.device}")
    if data.workers is not None:
        add(f"自对弈 worker 数: {data.workers}（持久进程池，全程复用）")
    if data.base_model:
        add(f"基础模型: {data.base_model}")
    if data.config:
        add("训练配置: " + ", ".join(f"{k}={v}" for k, v in data.config.items()))
    if data.evalset_games is not None or data.evalset_samples is not None:
        parts = []
        if data.evalset_games is not None:
            parts.append(f"{data.evalset_games} 局")
        if data.evalset_samples is not None:
            parts.append(f"增强后 {data.evalset_samples} 样本")
        add(f"固定评估集: {'，'.join(parts)}（不参与训练，跨轮次可比）")

    add("-" * 64)
    add("【完成情况】")
    if rounds:
        add(f"完整轮次: {len(rounds)} 轮（轮次 {rounds[0].step} ~ {rounds[-1].step}）")
        saved = [r for r in rounds if r.has_checkpoint]
        if saved:
            if len(saved) == 1:
                ckpt_desc = f"{saved[0].checkpoint}.pth"
            else:
                ckpt_desc = f"{saved[0].checkpoint}.pth ~ {saved[-1].checkpoint}.pth"
            add(f"已保存 checkpoint: {len(saved)} 个"
                f"（{ckpt_desc}，按 arena_interval 间隔保存；"
                f"其余轮次仅有训练/评估数据，未存 .pth）")
        else:
            add("已保存 checkpoint: 0 个（需训练满 arena_interval 轮才保存；"
                "下方各指标仍统计全部完整轮次）")
    if incomplete:
        steps = ", ".join(str(r.step) for r in incomplete)
        add(f"未完成轮次: {len(incomplete)} 轮（轮次 {steps}，日志结束时仍在进行）")
    if data.train_done_range:
        add(f"训练正常收尾: 是（轮次 {data.train_done_range[0]} ~ "
            f"{data.train_done_range[1]} 全部完成）")
    else:
        if rounds:
            add("训练正常收尾: 否（日志中未出现\"训练全部完成\"，可能被中断）")

    if not rounds:
        add("未解析到完整轮次，无法进行指标分析。")
        return "\n".join(lines)

    first, last = rounds[0], rounds[-1]

    # ---- 损失趋势（取每轮最后一个 epoch）----
    add("-" * 64)
    add("【损失趋势】（每轮最后一个 epoch 的值）")
    add(f"{'指标':<16}{'首轮(轮次' + str(first.step) + ')':>16}"
        f"{'末轮(轮次' + str(last.step) + ')':>16}{'历史最优(轮次)':>22}")

    def _row(name: str, getter):
        fv = getter(first.last_epoch)
        lv = getter(last.last_epoch)
        br, bv = _min_round(rounds, lambda r: getter(r.last_epoch))
        add(f"{name:<16}{fv:>16.4f}{lv:>16.4f}{bv:>14.4f} (轮次{br.step})")

    _row("train value", lambda e: e.train_value)
    _row("train policy", lambda e: e.train_policy)
    _row("val value", lambda e: e.val_value)
    _row("val policy", lambda e: e.val_policy)

    # val 总损失 = 2*val_value + val_policy（与训练损失加权口径一致）
    def _val_total(r: RoundStat) -> float:
        e = r.last_epoch
        return 2 * e.val_value + e.val_policy

    br, bv = _min_round(rounds, _val_total)
    add(f"{'val 总损失(2v+p)':<16}{_val_total(first):>16.4f}"
        f"{_val_total(last):>16.4f}{bv:>14.4f} (轮次{br.step})")

    # ---- 最佳模型（只能推荐实际落盘的 checkpoint）----
    saved = [r for r in rounds if r.has_checkpoint]
    sbr = None
    add("-" * 64)
    add("【最佳模型推荐】（仅在已保存的 checkpoint 中选取；"
        "checkpoint 按 arena_interval 间隔保存）")
    if data.best_step is not None:
        add(f"best.pth: val 滑动平均最优 {data.best_smooth:.4f}"
            f"（轮次 {data.best_step}，判据独立于 arena，持续覆盖更新）")
    if saved:
        sbr, sbv = _min_round(saved, _val_total)
        add(f"按 val 总损失: {sbr.checkpoint}.pth（轮次 {sbr.step}），"
            f"val 总损失 {sbv:.4f}，val value {sbr.last_epoch.val_value:.4f}，"
            f"val policy {sbr.last_epoch.val_policy:.4f}")
        sbr_v, sbv_v = _min_round(saved, lambda r: r.last_epoch.val_value)
        add(f"按 val value : {sbr_v.checkpoint}.pth（轮次 {sbr_v.step}），"
            f"val value {sbv_v:.4f}")
        sbr_p, sbv_p = _min_round(saved, lambda r: r.last_epoch.val_policy)
        add(f"按 val policy: {sbr_p.checkpoint}.pth（轮次 {sbr_p.step}），"
            f"val policy {sbv_p:.4f}")
    else:
        add("尚无已保存 checkpoint（需训练满 arena_interval 轮），"
            "暂无可推荐模型文件；损失趋势/固定评估仍可参考全部轮次。")

    # ---- 固定评估集（跨轮次可比）----
    fixed_rounds = [r for r in rounds if r.fixed_total is not None]
    if fixed_rounds:
        add("-" * 64)
        add("【固定评估集】（同一份数据跨轮次评估，可比；选模型主要依据）")
        ff, fl = fixed_rounds[0], fixed_rounds[-1]
        # 三个指标各自独立求 argmin（不能复用 total 的 argmin 轮次）
        br_v, best_v = _min_round(fixed_rounds, lambda r: r.fixed_value)
        br_p, best_p = _min_round(fixed_rounds, lambda r: r.fixed_policy)
        br_t, best_t = _min_round(fixed_rounds, lambda r: r.fixed_total)
        best_ckpt = (f"，checkpoint {br_t.checkpoint}.pth"
                     if br_t.has_checkpoint else "，该轮未保存 checkpoint")
        add(f"覆盖轮次: {ff.step} ~ {fl.step}（共 {len(fixed_rounds)} 轮）")
        add(f"固定评估 value : 首轮 {ff.fixed_value:.4f} → 末轮 {fl.fixed_value:.4f}"
            f"，最优 {best_v:.4f}（轮次 {br_v.step}）")
        add(f"固定评估 policy: 首轮 {ff.fixed_policy:.4f} → 末轮 {fl.fixed_policy:.4f}"
            f"，最优 {best_p:.4f}（轮次 {br_p.step}）")
        add(f"固定评估 总损失: 首轮 {ff.fixed_total:.4f} → 末轮 {fl.fixed_total:.4f}"
            f"，最优 {best_t:.4f}（轮次 {br_t.step}{best_ckpt}）")
        add("注: 每轮 val loss 来自不同自对弈棋局、跨轮次不可比；"
            "固定评估 loss 才是跨轮次可比指标。")

    # ---- 竞技场评估 ----
    if data.arenas:
        add("-" * 64)
        add("【竞技场评估】（新模型 vs 卫冕模型对弈，选模型的最终依据）")
        arenas = data.arenas
        promoted = [a for a in arenas if a.promoted]
        total_games = sum(a.games for a in arenas)
        add(f"对弈 {len(arenas)} 次共 {total_games} 局，晋升 {len(promoted)} 次")
        if promoted:
            hist = "、".join(f"轮次{a.step}" for a in promoted)
            cur = promoted[-1]
            add(f"晋升历史: {hist}")
            add(f"当前卫冕模型 champion.pth = 轮次 {cur.step}"
                f"（checkpoint {cur.step + 1}.pth，"
                f"对 champion(轮次{cur.champion_step}) 胜率 {cur.win_rate * 100:.1f}%）")
        else:
            add("当前卫冕模型仍为初始模型（尚无晋升）")
        rate_desc = (f"{data.arena_promote_rate:.0f}%"
                     if data.arena_promote_rate is not None else "配置值")
        add(f"晋升阈值: {rate_desc}")
        add("最近 5 次对弈:")
        for a in arenas[-5:]:
            tag = " → 晋升" if a.promoted else ""
            add(f"  轮次 {a.step:<4d} 对 champion(轮次{a.champion_step}): "
                f"{a.wins}胜{a.losses}负{a.draws}平，"
                f"胜率 {a.win_rate * 100:.1f}%{tag}")

    # ---- 泛化观察 ----
    add("-" * 64)
    add("【泛化 / 过拟合观察】")
    gap_first = first.last_epoch.val_value - first.last_epoch.train_value
    gap_last = last.last_epoch.val_value - last.last_epoch.train_value
    add(f"value 损失差值(val-train): 首轮 {gap_first:+.4f} → 末轮 {gap_last:+.4f}")
    bt, btv = _min_round(rounds, lambda r: r.last_epoch.train_value)
    brv, bvv = _min_round(rounds, lambda r: r.last_epoch.val_value)
    add(f"train value 最低点在轮次 {bt.step}（{btv:.4f}），"
        f"末轮为 {last.last_epoch.train_value:.4f}；"
        f"val value 最低点在轮次 {brv.step}（{bvv:.4f}）")
    if bv < _val_total(last) - 0.05:
        tip = (f"提示: val 总损失自轮次 {br.step} 后未再创新低，"
               f"继续训练的收益已明显变小，可考虑调整策略。")
        if sbr is not None:
            tip = (f"提示: val 总损失自轮次 {br.step} 后未再创新低，"
                   f"继续训练的收益已明显变小，可考虑使用 "
                   f"{sbr.checkpoint}.pth 或调整策略。")
        add(tip)

    # ---- 学习率 ----
    add("-" * 64)
    lr_rounds = [r for r in rounds if r.lr is not None]
    sm_rounds = [r for r in lr_rounds
                 if r.lr_phase is not None or r.lr_valley is not None]
    if sm_rounds:
        # 状态机（2026-09-20 定稿口径）：余弦下降 + 谷底停留 + arena 跳回
        add("【学习率调度】（余弦下降 + 谷底停留 + arena 驱动跳回，"
            "与评估信号解耦）")
        sr, sr_last = sm_rounds[0], sm_rounds[-1]
        add(f"学习率: {sr.lr:.2e} → {sr_last.lr:.2e}"
            f"（观测范围 {min(r.lr for r in sm_rounds):.2e} ~ "
            f"{max(r.lr for r in sm_rounds):.2e}）")
        desc = [r for r in sm_rounds if r.lr_phase is not None]
        valley = [r for r in sm_rounds if r.lr_valley is not None]
        if desc:
            cyc = desc[-1].lr_phase_len
            add(f"余弦下降段: 累计 {len(desc)} 轮（周期 {cyc} 轮），"
                f"最后观测相位 {desc[-1].lr_phase}/{cyc}")
        if valley:
            last_v = valley[-1]
            add(f"谷底停留: 累计 {len(valley)} 轮（当前第 {last_v.lr_valley} 轮，"
                f"arena 连续无晋升 {last_v.lr_fail}/{last_v.lr_fail_max}）")
        jumps = [(r.step, r.lr_jump_round) for r in sm_rounds
                 if r.lr_jump_round is not None]
        if jumps:
            add(f"跳回事件 {len(jumps)} 次: "
                + "、".join(f"r{s} 跳回，新周期自 r{j} 起" for s, j in jumps))
    elif lr_rounds:
        # ReduceLROnPlateau（run_6/7/8 历史日志）
        add("【学习率调度】（ReduceLROnPlateau：固定评估集总损失驱动，"
            "连续 patience 轮无改善则 lr × factor，下界 lr_min）")
        fr, lr_last = lr_rounds[0], lr_rounds[-1]
        add(f"学习率: {fr.lr:.2e} → {lr_last.lr:.2e}")
        best = min(r.lr_best for r in lr_rounds)
        pat = lr_last.lr_patience
        add(f"固定评估历史最优(best): {best:.4f}；"
            f"当前 bad_epochs {lr_last.lr_bad}/{pat}"
            + ("（即将触发降 lr）" if lr_last.lr_bad is not None
               and pat is not None and lr_last.lr_bad >= pat else ""))
        drops = []
        prev = None
        for r in lr_rounds:
            if prev is not None and r.lr < prev - 1e-30:
                drops.append(f"轮次{r.step}（{prev:.2e}→{r.lr:.2e}）")
            prev = r.lr
        add(f"共发生 {len(drops)} 次轮级降 lr")
        for d in drops[:10]:
            add(f"  - {d}")
        if len(drops) > 10:
            add(f"  ... 其余 {len(drops) - 10} 次略")
    else:
        # nn_012 方案（无轮级 [LR] 行）：每轮重建优化器，epoch 级 lr；
        # ReduceLROnPlateau 在轮内按验证损失下调（lr 每轮回峰值 learning_rate）
        add("【学习率调度】（nn_012 方案：每轮重建优化器 + "
            "ReduceLROnPlateau，epoch 级 lr 观测）")
        lr0 = first.epochs[0].lr
        lr_end = last.last_epoch.lr
        add(f"学习率: {lr0:.2e} → {lr_end:.2e}")
        all_lrs = [e.lr for r in rounds for e in r.epochs]
        if all_lrs:
            add(f"观测范围: 最低 {min(all_lrs):.2e}，最高 {max(all_lrs):.2e}，"
                f"共 {len(set(all_lrs))} 个不同取值")
        decays: list[str] = []
        prev_lr = None
        for r in rounds:
            for e in r.epochs:
                if prev_lr is not None and e.lr < prev_lr - 1e-30:
                    decays.append(
                        f"轮次{r.step} epoch{e.epoch}"
                        f"（{prev_lr:.2e}→{e.lr:.2e}）")
                prev_lr = e.lr
        add(f"共发生 {len(decays)} 次轮内 lr 下调"
            f"（ReduceLROnPlateau 触发，lr × factor）")
        for d in decays[:10]:
            add(f"  - {d}")
        if len(decays) > 10:
            add(f"  ... 其余 {len(decays) - 10} 次略")

    # ---- 自对弈数据量 ----
    add("-" * 64)
    add("【自对弈数据量】（回放池只存原始局；全节点收集后每局含大量"
        "中间局面样本，训练时取样在线随机 D4 增强）")
    sims = [r.num_sims for r in rounds if r.num_sims is not None]
    if sims:
        add(f"模拟次数/步: 首轮 {sims[0]}，末轮 {sims[-1]}"
            f"（最小 {min(sims)}，最大 {max(sims)}）")
    games = [r.replay_games for r in rounds if r.replay_games]
    if games:
        add(f"回放池局数: 首轮 {games[0]} → 末轮 {games[-1]}"
            f"（容量 {rounds[-1].replay_cap} 轮；"
            f"最小 {min(games)}，最大 {max(games)}）")
        if data.config.get("自对弈局数/轮"):
            n = data.config["自对弈局数/轮"]
            add(f"每轮新产自对弈: {n} 局（按局划分训练/验证）")
    if data.evalset_samples is not None:
        add(f"固定评估集: 增强后 {data.evalset_samples} 样本"
            + (f"（{data.evalset_games} 局）" if data.evalset_games else ""))

    # ---- 耗时 ----
    add("-" * 64)
    add("【耗时统计】")
    sp = [r.selfplay_sec for r in rounds if r.selfplay_sec is not None]
    tr = [r.train_sec for r in rounds if r.train_sec is not None]
    tt = [r.total_sec for r in rounds if r.total_sec is not None]
    add(f"自对弈阶段: 平均 {_avg(sp):.1f} 秒/轮，合计 {_fmt_dur(sum(sp))}")
    add(f"训练阶段  : 平均 {_avg(tr):.1f} 秒/轮，合计 {_fmt_dur(sum(tr))}")
    add(f"每轮总耗时: 平均 {_avg(tt):.1f} 秒，"
        f"最短 {_fmt_dur(min(tt))}，最长 {_fmt_dur(max(tt))}")
    if sp and data.config.get("自对弈局数/轮"):
        n = data.config["自对弈局数/轮"]
        workers_desc = (f"，{data.workers} worker 并行摊派"
                        if data.workers is not None else "")
        add(f"自对弈速度: 约 {_avg(sp) / n:.2f} 秒/局"
            f"（{_avg(sp):.1f} 秒/轮 ÷ {n} 局{workers_desc}）")

    if data.parse_warnings:
        add("-" * 64)
        add(f"【解析警告】共 {len(data.parse_warnings)} 条，前 5 条:")
        for w in data.parse_warnings[:5]:
            add(f"  {w}")

    add("=" * 64)
    return "\n".join(lines)


# ---------------------------------------------------------------- CSV 导出

def write_csv(data: LogData, out_path: Path) -> None:
    rounds = [r for r in data.rounds if r.completed]
    # 列数跟随日志真实 epoch 数（旧 3-epoch 日志出 e0~e2，新 1-epoch 日志
    # 只有 e0，避免出现整列无数据的空列）
    max_epochs = max((len(r.epochs) for r in rounds), default=0)

    header = ["轮次", "checkpoint", "模拟次数",
              "回放池填充", "回放池容量", "回放池局数"]
    for i in range(max_epochs):
        header += [f"e{i}_train_value", f"e{i}_train_policy",
                   f"e{i}_val_value", f"e{i}_val_policy", f"e{i}_lr"]
    header += [f"train_loss_e{i}" for i in range(max_epochs)]
    header += [f"val_loss_e{i}" for i in range(max_epochs)]
    header += ["自对弈耗时秒", "训练耗时秒", "总耗时秒",
               "固定评估_value", "固定评估_policy", "固定评估_总损失",
               "LR", "LR_相位", "LR_谷底轮", "LR_无晋升", "LR_best",
               "LR_bad_epochs",
               "竞技场_局数", "竞技场_胜", "竞技场_负", "竞技场_平",
               "竞技场_胜率", "竞技场_晋升", "模型路径"]

    arena_by_step = {a.step: a for a in data.arenas}
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rounds:
            row = [
                r.step, r.checkpoint,
                r.num_sims if r.num_sims is not None else "",
                r.replay_filled, r.replay_cap, r.replay_games,
            ]
            for i in range(max_epochs):
                if i < len(r.epochs):
                    e = r.epochs[i]
                    row += [f"{e.train_value:.4f}", f"{e.train_policy:.4f}",
                            f"{e.val_value:.4f}", f"{e.val_policy:.4f}",
                            f"{e.lr:.3e}"]
                else:
                    row += ["", "", "", "", ""]
            row += [f"{r.train_losses[i]:.4f}" if i < len(r.train_losses) else ""
                    for i in range(max_epochs)]
            row += [f"{r.val_losses[i]:.4f}" if i < len(r.val_losses) else ""
                    for i in range(max_epochs)]
            row += [
                f"{r.selfplay_sec:.1f}" if r.selfplay_sec is not None else "",
                f"{r.train_sec:.1f}" if r.train_sec is not None else "",
                f"{r.total_sec:.1f}" if r.total_sec is not None else "",
            ]
            row += [
                f"{r.fixed_value:.4f}" if r.fixed_value is not None else "",
                f"{r.fixed_policy:.4f}" if r.fixed_policy is not None else "",
                f"{r.fixed_total:.4f}" if r.fixed_total is not None else "",
                f"{r.lr:.3e}" if r.lr is not None else "",
                f"{r.lr_phase}/{r.lr_phase_len}"
                if r.lr_phase is not None else "",
                f"{r.lr_valley}" if r.lr_valley is not None else "",
                f"{r.lr_fail}/{r.lr_fail_max}"
                if r.lr_fail is not None else "",
                f"{r.lr_best:.4f}" if r.lr_best is not None else "",
                f"{r.lr_bad}/{r.lr_patience}" if r.lr_bad is not None else "",
            ]
            arena = arena_by_step.get(r.step)
            if arena is not None:
                row += [
                    arena.games, arena.wins, arena.losses, arena.draws,
                    f"{arena.win_rate:.3f}",
                    "是" if arena.promoted else "否",
                ]
            else:
                row += ["", "", "", "", "", ""]
            row.append(r.model_path)
            w.writerow(row)


# ---------------------------------------------------------------- 图表

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_losses(data: LogData, out_path: Path) -> None:
    plt = _setup_matplotlib()
    rounds = [r for r in data.rounds if r.completed and r.last_epoch]
    if not rounds:
        return
    xs = [r.step for r in rounds]
    tv = [r.last_epoch.train_value for r in rounds]
    vv = [r.last_epoch.val_value for r in rounds]
    tp = [r.last_epoch.train_policy for r in rounds]
    vp = [r.last_epoch.val_policy for r in rounds]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(xs, tv, label="train value", color="#1f77b4", lw=1.2)
    axes[0].plot(xs, vv, label="val value（每轮棋局不同）",
                 color="#d62728", lw=1.2)
    axes[1].plot(xs, tp, label="train policy", color="#1f77b4", lw=1.2)
    axes[1].plot(xs, vp, label="val policy（每轮棋局不同）",
                 color="#d62728", lw=1.2)

    # 固定评估集：跨轮次可比，虚线叠加
    fr = [r for r in rounds if r.fixed_value is not None]
    if fr:
        fx = [r.step for r in fr]
        axes[0].plot(fx, [r.fixed_value for r in fr],
                     label="固定评估 value（可比）", color="#9467bd",
                     lw=1.8, ls="--")
        axes[1].plot(fx, [r.fixed_policy for r in fr],
                     label="固定评估 policy（可比）", color="#9467bd",
                     lw=1.8, ls="--")

    axes[0].set_ylabel("value loss (MSE)")
    axes[0].set_title("Value 损失随训练轮次变化（每轮最后一个 epoch）")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("policy loss (KL)")
    axes[1].set_title("Policy 损失随训练轮次变化（每轮最后一个 epoch）")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_arena(data: LogData, out_path: Path) -> None:
    """竞技场胜率：柱形 + 晋升阈值线，晋升场次标绿。"""
    plt = _setup_matplotlib()
    arenas = data.arenas
    if not arenas:
        return
    xs = [a.step for a in arenas]
    rates = [a.win_rate * 100 for a in arenas]
    colors = ["#2ca02c" if a.promoted else "#9ecae1" for a in arenas]
    threshold = data.arena_promote_rate if data.arena_promote_rate is not None else 50.0

    # 柱宽随 arena 实际间距自适应（arena_interval 改动后不再失真）
    gaps = [b - a for a, b in zip(xs, xs[1:]) if b > a]
    width = (max(0.5, min(6.0, 0.8 * sorted(gaps)[len(gaps) // 2]))
             if gaps else 6.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(xs, rates, width=width, color=colors)
    ax.axhline(threshold, color="#d62728", ls="--", lw=1.2,
               label=f"晋升阈值 {threshold:.0f}%")
    ax.axhline(50, color="#999999", ls=":", lw=1, label="均势 50%")
    ax.set_title("竞技场胜率（新模型 vs 卫冕模型，绿色=晋升为新 champion）")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("胜率 (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lr_data(data: LogData, out_path: Path) -> None:
    """上图：学习率调度；下图：回放池局数（柱）+ 模拟次数（右轴折线）。"""
    plt = _setup_matplotlib()
    rounds = [r for r in data.rounds if r.completed and r.epochs]
    if not rounds:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 学习率：日志里有 [LR] 行就用轮级数据（X=训练轮次）；周期长度、
    # 谷底段、跳回点全部取自日志解析值（[LR] 余弦相位 n/N 的 N 即
    # lr_cycle_rounds），参数改动后图自动跟随，不硬编码。
    sm = [r for r in rounds if r.lr is not None]
    if sm:
        axes[0].semilogy([r.step for r in sm], [r.lr for r in sm],
                         color="#2ca02c", lw=1.2, drawstyle="steps-post")
        is_state = any(r.lr_phase is not None or r.lr_valley is not None
                       for r in sm)
        if is_state:
            # 周期边界：相位归零轮 + 跳回事件的新周期起始轮
            starts = sorted({r.step for r in sm if r.lr_phase == 0}
                            | {r.lr_jump_round for r in sm
                               if r.lr_jump_round is not None})
            for s in starts:
                axes[0].axvline(s, color="#999999", ls="--", lw=0.8,
                                alpha=0.7)
            # 谷底段（连续 lr_valley 区间）背景着色
            valley_xs = [r.step for r in sm if r.lr_valley is not None]
            if valley_xs:
                spans = []
                seg0 = prev = valley_xs[0]
                for s in valley_xs[1:]:
                    if s == prev + 1:
                        prev = s
                    else:
                        spans.append((seg0, prev))
                        seg0 = prev = s
                spans.append((seg0, prev))
                for a, b in spans:
                    axes[0].axvspan(a - 0.5, b + 0.5, color="#ff7f0e",
                                    alpha=0.12)
            cyc = max((r.lr_phase_len for r in sm
                       if r.lr_phase_len is not None), default=None)
            axes[0].set_title(
                "学习率调度（余弦下降+谷底停留〔橙底〕，竖线=新周期边界，"
                "arena 驱动跳回）" + (f"：周期 {cyc} 轮" if cyc else ""))
        else:
            axes[0].set_title("学习率调度（ReduceLROnPlateau，"
                              "固定评估集总损失驱动）")
        axes[0].set_xlabel("训练轮次")
        axes[0].set_ylabel("learning rate (log)")
        axes[0].grid(alpha=0.3, which="both")
    else:
        # 旧日志无 [LR] 行：退回 epoch 级展开；每轮 epoch 数从数据
        # 推断，配置行仅在缺失时兜底（不再写死 3）
        n_ep = (data.config.get("epoch")
                or max((len(r.epochs) for r in rounds), default=1))
        gx, ly = [], []
        for r in rounds:
            for e in r.epochs:
                gx.append(r.step * n_ep + e.epoch)
                ly.append(e.lr)
        axes[0].semilogy(gx, ly, color="#2ca02c", lw=1.2, marker=".",
                         markersize=3)
        axes[0].set_title("学习率调度（nn_012 方案：每轮重建优化器 + "
                          "ReduceLROnPlateau，epoch 级 lr）")
        axes[0].set_xlabel("全局 epoch")
        axes[0].set_ylabel("learning rate (log)")
        axes[0].grid(alpha=0.3, which="both")

    xs = [r.step for r in rounds]
    games = [r.replay_games for r in rounds]
    axes[1].bar(xs, games, label="回放池局数（原始局，未增强）",
                color="#9ecae1", width=1.0)
    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("回放池局数")
    axes[1].grid(alpha=0.3)

    sims = [(r.step, r.num_sims) for r in rounds if r.num_sims is not None]
    if sims:
        ax2 = axes[1].twinx()
        ax2.plot([s for s, _ in sims], [v for _, v in sims],
                 color="#ff7f0e", lw=1.5, label="模拟次数/步")
        ax2.set_ylabel("模拟次数/步", color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        axes[1].legend(loc="upper left")
    axes[1].set_title("回放池数据量与模拟次数")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_timing(data: LogData, out_path: Path) -> None:
    plt = _setup_matplotlib()
    rounds = [r for r in data.rounds
              if r.completed and r.selfplay_sec is not None and r.train_sec is not None]
    if not rounds:
        return
    xs = [r.step for r in rounds]
    sp = [r.selfplay_sec for r in rounds]
    tr = [r.train_sec for r in rounds]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(xs, sp, label="自对弈阶段", color="#9ecae1", width=1.0)
    ax.bar(xs, tr, bottom=sp, label="训练阶段", color="#fdae6b", width=1.0)
    ax.set_title("每轮耗时构成（自对弈：轮次开始→回放池行；训练：回放池行→val losses 行）")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("耗时（秒）")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程

def find_latest_log(log_dir: Path) -> Path:
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"{log_dir} 下没有 .log 文件")
    return logs[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="五子棋 CNN 训练日志分析")
    parser.add_argument("log", nargs="?", type=Path,
                        help="日志文件路径（默认取 logging/ 下最新）")
    args = parser.parse_args()

    log_path = args.log
    if log_path is None:
        log_path = find_latest_log(Path(__file__).parent / "logging")
    log_path = log_path.resolve()
    # log_path = Path(__file__).parent / "logging/2026_09_24__15_47_47.log"
    if not log_path.is_file():
        print(f"错误: 找不到日志文件 {log_path}", file=sys.stderr)
        return 1

    print(f"正在解析日志: {log_path}")
    data = parse_log(log_path)

    out_dir = log_path.parent / f"{log_path.stem}_分析"
    out_dir.mkdir(exist_ok=True)

    report = build_report(data)
    print()
    print(report)
    (out_dir / "分析报告.txt").write_text(report, encoding="utf-8")

    write_csv(data, out_dir / "每轮指标.csv")
    try:
        plot_losses(data, out_dir / "损失曲线.png")
        plot_lr_data(data, out_dir / "学习率与数据量.png")
        plot_timing(data, out_dir / "耗时分析.png")
        plot_arena(data, out_dir / "竞技场胜率.png")
        charts = "损失曲线.png / 学习率与数据量.png / 耗时分析.png / 竞技场胜率.png"
    except Exception as e:  # 图表失败不影响报告与 CSV
        charts = f"（图表生成失败: {e}）"

    print()
    print(f"分析完成，输出目录: {out_dir}")
    print(f"  - 分析报告.txt")
    print(f"  - 每轮指标.csv")
    print(f"  - {charts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
